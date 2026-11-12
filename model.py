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

# Step 5 - train
import torch

def train(model, data, block_size, batch_size, steps, lr=1e-3, generator=None):
    """AdamW training on random batches; returns each step's loss as a float."""

    # Create optimizer once
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []

    for step in range(steps):

        # Get a random batch
        xb, yb = get_batch(data, block_size, batch_size, generator)

        # Forward pass
        logits, loss = model(xb, yb)

        # Reset gradients
        optimizer.zero_grad(set_to_none=True)

        # Backpropagation
        loss.backward()

        # Update model parameters
        optimizer.step()

        # Store training loss as a Python float
        losses.append(loss.item())

    return losses

# Step 6 - estimate_loss
import torch

def estimate_loss(model, splits, block_size, batch_size, eval_iters, generator=None):
    """Mean loss over eval_iters batches for each named split, in evaluation mode, as floats."""

    results = {}

    # Switch model to evaluation mode
    model.eval()

    # Disable gradient tracking
    with torch.no_grad():

        # Iterate through each split in dictionary order
        for split, data in splits.items():

            losses = []

            for _ in range(eval_iters):

                # Get a random batch
                xb, yb = get_batch(data, block_size, batch_size, generator)

                # Calculate loss
                logits, loss = model(xb, yb)

                losses.append(loss.item())

            # Calculate mean loss for this split
            results[split] = sum(losses) / eval_iters

    # Restore training mode
    model.train()

    return results

# Step 7 - average_past
import torch
import torch.nn.functional as F

def average_past(x):
    """Position t of the output is the mean of x[:, :t + 1], computed with a masked softmax and a matrix multiply."""

    B, T, C = x.shape

    # Create a (T, T) weight matrix initialized with zeros
    weights = torch.zeros(T, T, device=x.device)

    # Mask positions above the diagonal with -inf
    weights = weights.masked_fill(
        torch.tril(torch.ones(T, T, device=x.device)) == 0,
        float('-inf')
    )

    # Apply softmax to get averaging weights
    weights = F.softmax(weights, dim=-1)

    # Multiply weights with x using batched matrix multiplication
    out = weights @ x

    return out

# Step 8 - Head
import torch
import torch.nn as nn
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, n_embd, head_size, block_size, dropout=0.0):
        """key, query and value projections, a registered tril buffer, and dropout on the attention weights."""
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # Causal mask: prevents looking at future tokens
        self.register_buffer(
            "tril",
            torch.tril(torch.ones(block_size, block_size))
        )

        # Dropout on attention weights
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """(B, T, n_embd) -> (B, T, head_size): scaled, causally masked attention."""

        B, T, C = x.shape

        # Generate keys and queries
        k = self.key(x)
        q = self.query(x)

        # Calculate scaled attention scores
        scores = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)

        # Apply causal mask
        scores = scores.masked_fill(
            self.tril[:T, :T] == 0,
            float('-inf')
        )

        # Convert scores into attention weights
        weights = F.softmax(scores, dim=-1)

        # Apply dropout
        weights = self.dropout(weights)

        # Weighted aggregation of values
        out = weights @ self.value(x)

        return out

# Step 9 - MultiHeadAttention
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size, n_embd, block_size, dropout=0.0):
        """num_heads Heads in an nn.ModuleList, a projection back to n_embd, and dropout."""
        super().__init__()

        # Create multiple independent attention heads
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size, dropout)
            for _ in range(num_heads)
        ])

        # Project concatenated output back to n_embd
        self.proj = nn.Linear(head_size * num_heads, n_embd)

        # Apply dropout after projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Concatenate the heads' outputs, project, dropout."""

        # Run all heads and concatenate along the last dimension
        out = torch.cat([head(x) for head in self.heads], dim=-1)

        # Projection followed by dropout
        out = self.dropout(self.proj(out))

        return out

# Step 10 - FeedForward
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        """Linear(n_embd, 4*n_embd) -> ReLU -> Linear(4*n_embd, n_embd) -> Dropout."""
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

# Step 11 - LayerNorm (not yet solved)
# TODO: implement

# Step 12 - Block (not yet solved)
# TODO: implement

# Step 13 - GPTLanguageModel (not yet solved)
# TODO: implement

