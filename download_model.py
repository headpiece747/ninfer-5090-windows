import os
import sys
import time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
from huggingface_hub import hf_hub_download

repo_id = "cometkim/Qwen3.8-27B-nvfp4qat-NInfer"
filename = "qwen3_8_27b_nvfp4qat.ninfer"
local_dir = r"C:\ai\models"

print(f"Starting ultra-fast download of {filename} from {repo_id} to {local_dir}...")
start_time = time.time()
try:
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
    )
except Exception as e:
    print(f"Error downloading: {e}")
    sys.exit(1)

elapsed = time.time() - start_time
size = os.path.getsize(path)
print(f"Download complete in {elapsed:.1f}s!")
print(f"File path: {path}")
print(f"File size: {size} bytes ({size / (1024**3):.2f} GiB)")
