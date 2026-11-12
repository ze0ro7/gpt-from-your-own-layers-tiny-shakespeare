"""
GPT From Your Own Layers: Tiny Shakespeare scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""GPT from your own layers, on Tiny Shakespeare.

Story: the bigram baseline from the start of the video, then a small GPT built from your Head,
MultiHeadAttention, FeedForward, LayerNorm and Block, trained on the same data with the same budget.
Compare their validation losses and read what each one writes.
(The video's full model, n_embd 384, 6 heads, 6 layers, block size 256, needs a GPU: about 1.48.)
"""
import torch


def main():
    chars, encode, decode, train_data, val_data = load_data()
    splits = {"train": train_data, "val": val_data}
    print(f"{len(train_data) + len(val_data):,} characters, vocabulary of {len(chars)}")

    # ---- 1. The bigram baseline ----
    torch.manual_seed(1337)
    bigram = BigramLanguageModel(len(chars))
    train(bigram, train_data, block_size=8, batch_size=32, steps=1500, lr=1e-2, generator=torch.Generator().manual_seed(0))
    est = estimate_loss(bigram, splits, 8, 32, 50, generator=torch.Generator().manual_seed(1))
    print("bigram  train {train:.3f}  val {val:.3f}".format(**est))

    # ---- 2. A small GPT ----
    block_size = 32
    torch.manual_seed(1337)
    gpt = GPTLanguageModel(len(chars), n_embd=64, n_head=4, n_layer=3, block_size=block_size, dropout=0.0)
    print(f"GPT parameters: {sum(p.numel() for p in gpt.parameters()):,}")
    train(gpt, train_data, block_size, batch_size=32, steps=1500, lr=2e-3, generator=torch.Generator().manual_seed(0))
    est = estimate_loss(gpt, splits, block_size, 32, 50, generator=torch.Generator().manual_seed(1))
    print("GPT     train {train:.3f}  val {val:.3f}".format(**est))

    # ---- 3. What they write ----
    start = torch.zeros((1, 1), dtype=torch.long)
    print("--- bigram ---")
    print(decode(generate(bigram, start, 200, 8, generator=torch.Generator().manual_seed(7))[0].tolist()))
    print("--- GPT ---")
    print(decode(generate(gpt, start, 300, block_size, generator=torch.Generator().manual_seed(7))[0].tolist()))


if __name__ == "__main__":
    main()

