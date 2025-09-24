import os
import shutil
import random

input_root = r"C:\Users\anton\datasets\PolyU_Cross_Iris"
output_root = r".\input_images"
max_images = None   # 🔹 Set how many images you want in the dataset (None = all)

# Make output directories
for modality in ["NIR", "VIS"]:
    os.makedirs(os.path.join(output_root, modality), exist_ok=True)

# Collect all files first
all_files = []
for root, _, files in os.walk(input_root):
    for file in files:
        # Accept both .tif and .tiff
        if file.lower().endswith((".tif", ".tiff")):
            all_files.append((root, file))

# Shuffle the list
random.shuffle(all_files)

# Take only the requested number
selected_files = all_files[:max_images] if max_images is not None else all_files

count = 0
for root, file in selected_files:
    parts = file.split("_")
    if len(parts) < 4:
        continue
    
    individual = parts[0]                  # "001"
    side = parts[1]                        # "L" or "R"
    modality = parts[2]                    # "NIR" or "VIS"
    timestamp = os.path.splitext(parts[3])[0]
    
    side_code = "0" if side.upper() == "L" else "1"
    timestamp = timestamp.zfill(2)
    
    # Always rename to .tiff
    new_filename = f"{individual}{side_code}_{timestamp}.tiff"
    dst = os.path.join(output_root, modality, new_filename)
    
    shutil.copy2(os.path.join(root, file), dst)
    count += 1
    print(f"[{count}] Copied {file} -> {dst}")

print(f"\n✅ Done! {count} images copied randomly into dataset/")
