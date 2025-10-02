import os
import shutil
import subprocess
from pathlib import Path
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv(".env")

# --- Configuration (SEGMENTER) ---
SEGMENTER_CONTAINER_NAME = os.getenv("SEGMENTER_CONTAINER_NAME")
SEGMENTER_IMAGE = os.getenv("SEGMENTER_IMAGE")
SEGMENTER_FOLDER = os.getenv("SEGMENTER_FOLDER")
SEGMENTER_INPUT_FOLDER = os.getenv("SEGMENTER_INPUT_FOLDER")
SEGMENTER_OUTPUT_FOLDER = os.getenv("SEGMENTER_OUTPUT_FOLDER")
SEGMENTER_COMMAND = os.getenv("SEGMENTER_COMMAND")

# --- Configuration (MATCHER: includes encoding + matching + stats) ---
MATCHER_CONTAINER_NAME = os.getenv("MATCHER_CONTAINER_NAME")      # 'iris_matcher'
MATCHER_IMAGE = os.getenv("MATCHER_IMAGE")                        # 'antoniounimi/usit'
MATCHER_FOLDER = os.getenv("MATCHER_FOLDER")                      # 'usit'
MATCHER_INPUT_IMAGES = os.getenv("MATCHER_INPUT_IMAGES")          # 'data/input_images'
MATCHER_INPUT_MASKS = os.getenv("MATCHER_INPUT_MASKS")            # 'data/masks'
MATCHER_OUTPUT_FOLDER = os.getenv("MATCHER_OUTPUT_FOLDER")        # 'data/distances'
MATCHER_COMMAND = os.getenv("MATCHER_COMMAND")                    # encode+match+stats (chained)

# --- Paths (host side) ---
BASE_DIR = os.getcwd()
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "DB_SYNTIRIS_vero_0_5_selectedframes")   # your source images
LOCAL_MASKS_DIR = os.path.join(BASE_DIR, "masks")           # masks produced by segmenter
LOCAL_DISTANCES_DIR = os.path.join(BASE_DIR, "distances")   # matcher results

# Segmenter mount root (host path)
SEGMENTER_PATH = os.path.join(BASE_DIR, SEGMENTER_FOLDER)

# Matcher mount root (host path)
MATCHER_PATH = os.path.join(BASE_DIR, MATCHER_FOLDER)

# --- Helpers ---
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_all_files(src_dir: str, dst_dir: str):
    if not os.path.isdir(src_dir):
        print(f"[WARN] Source directory does not exist, skipping copy: {src_dir}")
        return
    for root, _, files in os.walk(src_dir):
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        ensure_dir(target_root)
        for fname in files:
            shutil.copy2(os.path.join(root, fname), os.path.join(target_root, fname))

def run_docker_once(name: str, image: str, host_mount: str, workdir_in_container: str, command: str):
    docker_cmd = [
        "docker", "run", "-it", "--rm",
        "--name", name,
        "-v", f"{host_mount}:/host",
        "-w", workdir_in_container,
        image,
        "bash", "-ic", command
    ]
    print(f"\n[DOCKER] Running: {' '.join(docker_cmd)}\n")
    subprocess.run(docker_cmd, check=True)

# --- Sanity checks ---
required_env = {
    "SEGMENTER_CONTAINER_NAME": SEGMENTER_CONTAINER_NAME,
    "SEGMENTER_IMAGE": SEGMENTER_IMAGE,
    "SEGMENTER_FOLDER": SEGMENTER_FOLDER,
    "SEGMENTER_INPUT_FOLDER": SEGMENTER_INPUT_FOLDER,
    "SEGMENTER_OUTPUT_FOLDER": SEGMENTER_OUTPUT_FOLDER,
    "SEGMENTER_COMMAND": SEGMENTER_COMMAND,
    "MATCHER_CONTAINER_NAME": MATCHER_CONTAINER_NAME,
    "MATCHER_IMAGE": MATCHER_IMAGE,
    "MATCHER_FOLDER": MATCHER_FOLDER,
    "MATCHER_INPUT_IMAGES": MATCHER_INPUT_IMAGES,
    "MATCHER_INPUT_MASKS": MATCHER_INPUT_MASKS,
    "MATCHER_OUTPUT_FOLDER": MATCHER_OUTPUT_FOLDER,
    "MATCHER_COMMAND": MATCHER_COMMAND,
}
missing = [k for k, v in required_env.items() if not v]
if missing:
    raise RuntimeError(f"Missing required .env variables: {', '.join(missing)}")

# --- Ensure local folders exist ---
ensure_dir(SEGMENTER_PATH)
ensure_dir(os.path.join(SEGMENTER_PATH, SEGMENTER_INPUT_FOLDER))
ensure_dir(LOCAL_MASKS_DIR)

ensure_dir(MATCHER_PATH)
ensure_dir(os.path.join(MATCHER_PATH, MATCHER_INPUT_IMAGES))
ensure_dir(os.path.join(MATCHER_PATH, MATCHER_INPUT_MASKS))
ensure_dir(os.path.join(MATCHER_PATH, MATCHER_OUTPUT_FOLDER))
ensure_dir(LOCAL_DISTANCES_DIR)

# =========================
# STEP 1: SEGMENTER
# =========================
print("[STEP 1] Preparing inputs for segmenter...")
copy_all_files(INPUT_IMAGES_DIR, os.path.join(SEGMENTER_PATH, SEGMENTER_INPUT_FOLDER))
print(f"  → Copied input images to {SEGMENTER_FOLDER}/{SEGMENTER_INPUT_FOLDER}/")

print("[STEP 1] Running segmentation container...")
run_docker_once(
    name=SEGMENTER_CONTAINER_NAME,
    image=SEGMENTER_IMAGE,
    host_mount=SEGMENTER_PATH,
    workdir_in_container="/host",
    command=SEGMENTER_COMMAND
)
print("[STEP 1] Segmentation finished.")

print("[STEP 1] Collecting masks locally...")
container_output_path = os.path.join(SEGMENTER_PATH, SEGMENTER_OUTPUT_FOLDER)
copy_all_files(container_output_path, LOCAL_MASKS_DIR)
print(f"  → Results copied to {LOCAL_MASKS_DIR}/")

# =========================
# STEP 2: MATCHER (encode + match + stats)
# =========================
print("\n[STEP 2] Preparing inputs for matcher...")

matcher_images_mount = os.path.join(MATCHER_PATH, MATCHER_INPUT_IMAGES)
matcher_masks_mount = os.path.join(MATCHER_PATH, MATCHER_INPUT_MASKS)

# Copy original images and freshly produced masks
copy_all_files(INPUT_IMAGES_DIR, matcher_images_mount)
copy_all_files(LOCAL_MASKS_DIR, matcher_masks_mount)
print(f"  → Copied images to {MATCHER_FOLDER}/{MATCHER_INPUT_IMAGES}/")
print(f"  → Copied masks to  {MATCHER_FOLDER}/{MATCHER_INPUT_MASKS}/")

print("[STEP 2] Running matcher container (encode + match + stats)...")
run_docker_once(
    name=MATCHER_CONTAINER_NAME,
    image=MATCHER_IMAGE,
    host_mount=MATCHER_PATH,
    workdir_in_container="/host",
    command=MATCHER_COMMAND
)
print("[STEP 2] Matcher finished.")

print("\n[STEP 2] Collecting matcher outputs...")
matcher_output_path = os.path.join(MATCHER_PATH, MATCHER_OUTPUT_FOLDER)
copy_all_files(matcher_output_path, LOCAL_DISTANCES_DIR)
print(f"  → distances/results saved to {LOCAL_DISTANCES_DIR}/")

print("\n✅ Pipeline complete (2 steps: Segmenter, Matcher).")

