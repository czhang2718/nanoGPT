import os
import sys
os.environ["HF_HOME"] = "/data/scratch-oc40/htfan/hf_home"
os.environ["PYTHONUNBUFFERED"] = "1"

import tiktoken
import numpy as np
from datasets import load_dataset

# Load the dataset
print("Loading OpenWebText dataset...", flush=True)
ds = load_dataset("skylion007/openwebtext", split="train")

# Encode with tiktoken gpt2 bpe
print("Tokenizing with GPT-2 BPE...", flush=True)
enc = tiktoken.get_encoding("gpt2")

# Process in chunks to avoid memory issues - tokenize as we go
all_ids = []
for i, example in enumerate(ds):
    text = example['text']
    ids = enc.encode_ordinary(text)
    all_ids.extend(ids)
    if (i + 1) % 100000 == 0:
        print(f"Processed {i + 1:,} documents, {len(all_ids):,} tokens so far...", flush=True)

print(f"Total tokens: {len(all_ids):,}", flush=True)

# Split into train and val (99% train, 1% val for large datasets)
n = len(all_ids)
split_idx = int(n * 0.99)
train_ids = all_ids[:split_idx]
val_ids = all_ids[split_idx:]

print(f"train has {len(train_ids):,} tokens", flush=True)
print(f"val has {len(val_ids):,} tokens", flush=True)

# Export to bin files
print("Saving to binary files...", flush=True)
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

print("Done! Files saved:", flush=True)
print(f"  - {os.path.join(os.path.dirname(__file__), 'train.bin')}", flush=True)
print(f"  - {os.path.join(os.path.dirname(__file__), 'val.bin')}", flush=True)