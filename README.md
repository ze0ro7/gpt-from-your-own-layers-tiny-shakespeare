# GPT From Your Own Layers: Tiny Shakespeare

Follow Andrej Karpathy's "Let's build GPT" and write a character-level GPT in PyTorch, one layer at a time. Load Tiny Shakespeare with a character tokenizer, batch random chunks, and set up a bigram baseline with generate, train and estimate-loss functions. Then build self-attention from the masked weighted average of the past: a causal Head with scaled dot-product attention, multi-head attention with a projection, and a feed-forward layer. Finish with your own LayerNorm, a pre-norm residual transformer Block, and the full GPT with token and position embeddings, and compare what it writes with the bigram's.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** load_data
- [x] **2.** get_batch
- [x] **3.** BigramLanguageModel
- [x] **4.** generate
- [x] **5.** train
- [x] **6.** estimate_loss
- [x] **7.** average_past
- [x] **8.** Head
- [x] **9.** MultiHeadAttention
- [x] **10.** FeedForward
- [x] **11.** LayerNorm
- [x] **12.** Block
- [x] **13.** GPTLanguageModel

---

Built on Deep-ML.
