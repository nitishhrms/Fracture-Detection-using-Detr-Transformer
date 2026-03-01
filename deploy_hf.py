"""
deploy_hf.py — Upload project to HuggingFace Spaces
Run this AFTER: python -c "from huggingface_hub import login; login()"
"""
from huggingface_hub import HfApi, create_repo
import os

HF_USERNAME  = "nitishhrms"           # your HF username
SPACE_NAME   = "fracture-detection-detr"
SPACE_REPO   = f"{HF_USERNAME}/{SPACE_NAME}"
SPACE_DIR    = "huggingface_space"    # local folder to upload

api = HfApi()

# 1. Create the Space (skips if already exists)
print(f"Creating Space: {SPACE_REPO} ...")
try:
    create_repo(
        repo_id=SPACE_REPO,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
        private=False,
    )
    print(f"Space ready: https://huggingface.co/spaces/{SPACE_REPO}")
except Exception as e:
    print(f"Repo create note: {e}")

# 2. Upload all files from huggingface_space/
print("\nUploading files...")
for fname in os.listdir(SPACE_DIR):
    fpath = os.path.join(SPACE_DIR, fname)
    if os.path.isfile(fpath):
        api.upload_file(
            path_or_fileobj=fpath,
            path_in_repo=fname,
            repo_id=SPACE_REPO,
            repo_type="space",
        )
        print(f"  ✓ {fname}")

print(f"\nDone! View your Space at:")
print(f"  https://huggingface.co/spaces/{SPACE_REPO}")
print(f"\nIt takes ~2 minutes to build. Refresh the page after that.")
