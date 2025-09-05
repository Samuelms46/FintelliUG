import os
from config import Config

persist_dir = Config.VECTOR_DB_PATH  # or CHROMA_PERSIST_DIR

# Create the directory if it doesn't exist
os.makedirs(persist_dir, exist_ok=True)

if os.access(persist_dir, os.W_OK):
    print(f"Directory '{persist_dir}' exists and is writable.")
else:
    print(f"Directory '{persist_dir}' is not writable!")