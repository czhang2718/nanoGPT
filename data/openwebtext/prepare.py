import os
os.environ["HF_HOME"] = "/data/scratch-oc40/htfan/hf_home"

from datasets import load_dataset

ds = load_dataset("skylion007/openwebtext", split="train")
