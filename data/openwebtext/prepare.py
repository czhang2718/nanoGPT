import os
os.environ["HF_HOME"] = "/work1/sawyma/czhang/data"

from datasets import load_dataset

ds = load_dataset("skylion007/openwebtext", split="train")
