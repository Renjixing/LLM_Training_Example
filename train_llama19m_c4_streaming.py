import argparse
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, Optional

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from transformers import (
    AutoTokenizer,
    LlamaConfig,
    LlamaForCausalLM,
    get_linear_schedule_with_warmup,
)


@dataclass
class GradStats:
    total_grad_elements: int = 0
    total_grad_bytes: int = 0


class C4TokenStreamDataset(IterableDataset):
    """将 C4 文本流式读取并分块为固定长度 token。"""

    def __init__(
        self,
        tokenizer,
        split: str,
        max_seq_len: int,
        target_tokens: Optional[int] = None,
        buffer_texts: int = 64,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        self.split = split
        self.max_seq_len = max_seq_len
        self.target_tokens = target_tokens
        self.buffer_texts = buffer_texts
        self.seed = seed

    def _get_stream(self):
        stream = load_dataset("allenai/c4", "en", split=self.split, streaming=True)
        if self.split == "train":
            stream = stream.shuffle(buffer_size=10000, seed=self.seed)
        return stream

    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        token_buffer = []
        emitted_tokens = 0
        stream = self._get_stream()

        for sample in stream:
            text = sample.get("text", "")
            if not text.strip():
                continue
            ids = self.tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids + [self.tokenizer.eos_token_id])

            while len(token_buffer) >= self.max_seq_len + 1:
                if self.target_tokens is not None and emitted_tokens >= self.target_tokens:
                    return

                chunk = token_buffer[: self.max_seq_len + 1]
                token_buffer = token_buffer[self.max_seq_len + 1 :]

                input_ids = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:], dtype=torch.long)
                emitted_tokens += input_ids.numel()

                yield {
                    "input_ids": input_ids,
                    "labels": labels,
                }

                if self.target_tokens is not None and emitted_tokens >= self.target_tokens:
                    return


def build_llama_19m(vocab_size: int) -> LlamaForCausalLM:
    """构建约 19M 参数量的 LLaMA 模型。"""
    config = LlamaConfig(
        vocab_size=vocab_size,
        hidden_size=384,
        intermediate_size=1536,
        num_hidden_layers=12,
        num_attention_heads=6,
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = LlamaForCausalLM(config)
    return model


def collate_fn(batch):
    input_ids = torch.stack([x["input_ids"] for x in batch])
    labels = torch.stack([x["labels"] for x in batch])
    return {"input_ids": input_ids, "labels": labels}


def count_grad_stats(model: torch.nn.Module) -> GradStats:
    grad_elems = 0
    grad_bytes = 0
    for p in model.parameters():
        if p.grad is not None:
            grad_elems += p.grad.numel()
            grad_bytes += p.grad.numel() * p.grad.element_size()
    return GradStats(total_grad_elements=grad_elems, total_grad_bytes=grad_bytes)


@torch.no_grad()
def evaluate_perplexity(model, dataloader, device, max_eval_batches: int = 200) -> float:
    model.eval()
    losses = []
    for i, batch in enumerate(dataloader):
        if i >= max_eval_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        losses.append(out.loss.item())
    mean_loss = sum(losses) / max(len(losses), 1)
    ppl = float(math.exp(mean_loss))
    model.train()
    return ppl


def train(args):
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = build_llama_19m(vocab_size=len(tokenizer))
    model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    target_train_tokens = args.target_multiplier * num_params

    train_ds = C4TokenStreamDataset(
        tokenizer=tokenizer,
        split="train",
        max_seq_len=args.seq_len,
        target_tokens=target_train_tokens,
        seed=args.seed,
    )
    val_ds = C4TokenStreamDataset(
        tokenizer=tokenizer,
        split="validation",
        max_seq_len=args.seq_len,
        target_tokens=args.eval_tokens,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    steps_per_epoch = max(
        1,
        target_train_tokens // (args.batch_size * args.seq_len * args.grad_accum_steps),
    )
    total_steps = steps_per_epoch * args.epochs

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )

    global_step = 0
    consumed_tokens = 0
    total_grad_elems = 0
    total_grad_bytes = 0

    for epoch in range(args.epochs):
        for step, batch in enumerate(train_loader):
            if global_step >= total_steps:
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.grad_accum_steps
            loss.backward()

            consumed_tokens += batch["input_ids"].numel()

            if (step + 1) % args.grad_accum_steps == 0:
                grad_stats = count_grad_stats(model)
                total_grad_elems += grad_stats.total_grad_elements
                total_grad_bytes += grad_stats.total_grad_bytes

                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_steps == 0:
                    print(
                        f"epoch={epoch} step={global_step}/{total_steps} "
                        f"loss={loss.item() * args.grad_accum_steps:.4f} "
                        f"tokens={consumed_tokens} "
                        f"grad_elems={total_grad_elems} "
                        f"grad_MB={total_grad_bytes / 1024**2:.2f}"
                    )

                if global_step % args.eval_steps == 0:
                    ppl = evaluate_perplexity(
                        model,
                        val_loader,
                        device,
                        max_eval_batches=args.max_eval_batches,
                    )
                    print(f"[Eval] step={global_step} perplexity={ppl:.3f}")

        if global_step >= total_steps:
            break

    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    final_ppl = evaluate_perplexity(model, val_loader, device, args.max_eval_batches)
    print("=" * 80)
    print(f"模型参数量: {num_params:,}")
    print(f"目标训练 tokens (4x 参数量默认): {target_train_tokens:,}")
    print(f"实际消耗 tokens: {consumed_tokens:,}")
    print(f"累计梯度元素总量: {total_grad_elems:,}")
    print(f"累计梯度数据总量: {total_grad_bytes / 1024**3:.4f} GB")
    print(f"最终验证集 Perplexity: {final_ppl:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C4 流式训练 LLaMA-19M 示例")
    parser.add_argument("--tokenizer_name", type=str, default="hf-internal-testing/llama-tokenizer")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/llama19m_c4")
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--grad_accum_steps", type=int, default=4)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--log_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--max_eval_batches", type=int, default=100)
    parser.add_argument("--eval_tokens", type=int, default=2_000_000)
    parser.add_argument("--target_multiplier", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train(args)
