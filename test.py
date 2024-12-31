import os
import shutil

source_dir = "datasets/dataset-50k/midi"
target_dir = "bad-test"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Files to copy
files_to_copy = [f"{i:05d}_midi.mid" for i in range(3460, 3470)]

# Copy each file
for filename in files_to_copy:
    src = os.path.join(source_dir, filename)
    if os.path.exists(src):
        dst = os.path.join(target_dir, filename)
        shutil.copy2(src, dst)
