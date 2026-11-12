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

# Step 2 - get_batch (not yet solved)
# TODO: implement

# Step 3 - BigramLanguageModel (not yet solved)
# TODO: implement

# Step 4 - generate (not yet solved)
# TODO: implement

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

