"""
GPT From Your Own Layers: Tiny Shakespeare

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - load_data
import os
import tempfile
import urllib.request
import torch

def load_data(train_frac=0.9):
    """Return (chars, encode, decode, train_data, val_data) for Tiny Shakespeare, one id per character."""

    # Download the dataset only if it doesn't already exist
    file_path = os.path.join(tempfile.gettempdir(), "tinyshakespeare.txt")
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"

    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)

    # Read the text
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Get sorted unique characters
    chars = sorted(list(set(text)))

    # Create character-to-ID and ID-to-character mappings
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    # Encoding and decoding functions
    def encode(s):
        return [stoi[ch] for ch in s]

    def decode(ids):
        return "".join(itos[i] for i in ids)

    # Encode the entire text into a 1-D torch.long tensor
    data = torch.tensor(encode(text), dtype=torch.long)

    # Split into training and validation data
    split_idx = int(train_frac * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    return chars, encode, decode, train_data, val_data

# Step 2 - get_batch
import torch

def get_batch(data, block_size, batch_size, generator=None):
    """Random (batch_size, block_size) chunks x of data and their next-token targets y."""

    # Generate random starting positions for each batch
    offsets = torch.randint(
        len(data) - block_size,
        (batch_size,),
        generator=generator
    )

    # Input sequences
    x = torch.stack([
        data[i : i + block_size]
        for i in offsets
    ])

    # Target sequences (shifted by one position)
    y = torch.stack([
        data[i + 1 : i + block_size + 1]
        for i in offsets
    ])

    return x, y

# Step 3 - BigramLanguageModel
import torch
import torch.nn as nn
import torch.nn.functional as F

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        """One embedding table: row i holds the logits for the token after token i."""
        super().__init__()

        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """Return (logits of shape (B, T, vocab_size), loss or None)."""

        # Look up embeddings to get logits
        logits = self.token_embedding_table(idx)

        loss = None

        if targets is not None:
            B, T, C = logits.shape

            # Flatten logits and targets for cross entropy
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)

            loss = F.cross_entropy(logits, targets)

            # Reshape logits back to original shape
            logits = logits.view(B, T, C)

        return logits, loss

# Step 4 - generate
import torch
import torch.nn.functional as F

def generate(model, idx, max_new_tokens, block_size, generator=None):
    """Append max_new_tokens sampled tokens to idx (B, T), cropping the context to block_size."""

    # Disable gradient tracking during generation
    with torch.no_grad():

        for _ in range(max_new_tokens):

            # Crop context to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # Get predictions from the model
            logits, _ = model(idx_cond)

            # Keep only the logits for the last time step
            logits = logits[:, -1, :]

            # Convert logits into probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample the next token for each batch row
            idx_next = torch.multinomial(
                probs,
                num_samples=1,
                generator=generator
            )

            # Append the sampled token to the sequence
            idx = torch.cat((idx, idx_next), dim=1)

    return idx

# Step 5 - train (not yet solved)
# TODO: implement

# Step 6 - estimate_loss (not yet solved)
# TODO: implement

# Step 7 - average_past (not yet solved)
# TODO: implement

# Step 8 - Head (not yet solved)
# TODO: implement

# Step 9 - MultiHeadAttention (not yet solved)
# TODO: implement

# Step 10 - FeedForward (not yet solved)
# TODO: implement

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

