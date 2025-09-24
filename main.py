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

# --- Configuration (ENCODER) ---
ENCODER_CONTAINER_NAME = os.getenv("ENCODER_CONTAINER_NAME")  # e.g. 'iris_encoder'
ENCODER_IMAGE = os.getenv("ENCODER_IMAGE")                    # e.g. 'antoniounimi/usit'
ENCODER_FOLDER = os.getenv("ENCODER_FOLDER")                  # e.g. 'usit'
ENCODER_INPUT_IMAGES = os.getenv("ENCODER_INPUT_IMAGES")      # e.g. 'data/input_images'
ENCODER_INPUT_MASKS  = os.getenv("ENCODER_INPUT_MASKS")       # e.g. 'data/masks'
ENCODER_COMMAND = os.getenv("ENCODER_COMMAND")                # e.g. 'python3 run.py'

# --- Configuration (MATCHER) ---
# Defaults are set to the values you specified; you can still override via .env if desired.
MATCHER_CONTAINER_NAME = os.getenv("MATCHER_CONTAINER_NAME", "iris_matcher")
MATCHER_IMAGE = os.getenv("MATCHER_IMAGE", "antoniounimi/usit")
MATCHER_FOLDER = os.getenv("MATCHER_FOLDER", "usit")
MATCHER_OUTPUT_FOLDER = os.getenv("MATCHER_OUTPUT_FOLDER", "data/distances")
MATCHER_COMMAND = os.getenv(
    "MATCHER_COMMAND",
    "python matcher.py && python gen_stats_np.py -i data/distances/hamming_distance.txt -o data/distances/results.txt -id '(\\d{4})'"
)

# --- Paths (host side) ---
BASE_DIR = os.getcwd()
INPUT_IMAGES_DIR = os.path.join(BASE_DIR, "input_images")   # your source images
LOCAL_MASKS_DIR = os.path.join(BASE_DIR, "masks")           # masks produced by segmenter (and used for encoder)
LOCAL_IRISCODE_DIR = os.path.join(BASE_DIR, "irisCode")     # where we'll gather encoder output (iris codes)
LOCAL_IRISCODE_MASKS_DIR = os.path.join(BASE_DIR, "irisCode_masks")
LOCAL_DISTANCES_DIR = os.path.join(BASE_DIR, "distances")   # where we'll gather matcher output (distances, results)

# Segmenter mount root (host path)
SEGMENTER_PATH = os.path.join(BASE_DIR, SEGMENTER_FOLDER)

# Encoder mount root (host path)
ENCODER_PATH = os.path.join(BASE_DIR, ENCODER_FOLDER)

# Matcher mount root (host path)
MATCHER_PATH = os.path.join(BASE_DIR, MATCHER_FOLDER)

# --- Helpers ---
def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

def copy_all_files(src_dir: str, dst_dir: str):
    if not os.path.isdir(src_dir):
        print(f"[WARN] Source directory does not exist, skipping copy: {src_dir}")
        return
    
    for root, dirs, files in os.walk(src_dir):
        # Recreate the directory structure in dst_dir
        rel_path = os.path.relpath(root, src_dir)
        target_root = os.path.join(dst_dir, rel_path)
        ensure_dir(target_root)

        # Copy all files
        for fname in files:
            s = os.path.join(root, fname)
            d = os.path.join(target_root, fname)
            shutil.copy2(s, d)

def run_docker_once(name: str, image: str, host_mount: str, workdir_in_container: str, command: str):
    docker_cmd = [
        "docker", "run", "-it", "--rm",
        "--name", name,
        "-v", f"{host_mount}:/host",
        "-w", workdir_in_container,   # usually "/host"
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
    "ENCODER_CONTAINER_NAME": ENCODER_CONTAINER_NAME,
    "ENCODER_IMAGE": ENCODER_IMAGE,
    "ENCODER_FOLDER": ENCODER_FOLDER,
    "ENCODER_INPUT_IMAGES": ENCODER_INPUT_IMAGES,
    "ENCODER_INPUT_MASKS": ENCODER_INPUT_MASKS,
    "ENCODER_COMMAND": ENCODER_COMMAND,
    # Matcher is also required (we have sensible defaults, but you can still override via .env)
    "MATCHER_CONTAINER_NAME": MATCHER_CONTAINER_NAME,
    "MATCHER_IMAGE": MATCHER_IMAGE,
    "MATCHER_FOLDER": MATCHER_FOLDER,
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

ensure_dir(ENCODER_PATH)
ensure_dir(os.path.join(ENCODER_PATH, ENCODER_INPUT_IMAGES))
ensure_dir(os.path.join(ENCODER_PATH, ENCODER_INPUT_MASKS))
ensure_dir(LOCAL_IRISCODE_DIR)
ensure_dir(LOCAL_IRISCODE_MASKS_DIR)

ensure_dir(MATCHER_PATH)
ensure_dir(os.path.join(MATCHER_PATH, MATCHER_OUTPUT_FOLDER))
ensure_dir(LOCAL_DISTANCES_DIR)

# =========================
# STEP 1: Prepare & run SEGMENTER
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

# Pull masks from segmenter mount to local masks folder
print("[STEP 1] Collecting masks locally...")
container_output_path = os.path.join(SEGMENTER_PATH, SEGMENTER_OUTPUT_FOLDER)
copy_all_files(container_output_path, LOCAL_MASKS_DIR)
print(f"  → Results copied to {LOCAL_MASKS_DIR}/")

# =========================
# STEP 2: Prepare & run ENCODER
# =========================
print("\n[STEP 2] Preparing inputs for encoder...")

# The encoder expects:
#  - images under ENCODER_INPUT_IMAGES (e.g. usit/data/input_images)
#  - masks under ENCODER_INPUT_MASKS (e.g. usit/data/masks)
encoder_images_mount = os.path.join(ENCODER_PATH, ENCODER_INPUT_IMAGES)
encoder_masks_mount = os.path.join(ENCODER_PATH, ENCODER_INPUT_MASKS)

# Copy host-side original images and the freshly produced masks into the encoder's tree
copy_all_files(INPUT_IMAGES_DIR, encoder_images_mount)
copy_all_files(LOCAL_MASKS_DIR, encoder_masks_mount)
print(f"  → Copied images to {ENCODER_FOLDER}/{ENCODER_INPUT_IMAGES}/")
print(f"  → Copied masks to  {ENCODER_FOLDER}/{ENCODER_INPUT_MASKS}/")

print("[STEP 2] Running encoder container...")
run_docker_once(
    name=ENCODER_CONTAINER_NAME,
    image=ENCODER_IMAGE,
    host_mount=ENCODER_PATH,
    workdir_in_container="/host",
    command=ENCODER_COMMAND
)
print("[STEP 2] Encoding finished.")

# =========================
# STEP 3: Collect ENCODER outputs to local folders
# =========================
print("\n[STEP 3] Collecting encoder outputs...")

# The run.py writes to usit/data/irisCode and usit/data/irisCode_masks
encoder_iriscode_path = os.path.join(ENCODER_PATH, "data", "irisCode")
encoder_iriscode_masks_path = os.path.join(ENCODER_PATH, "data", "irisCode_masks")

copy_all_files(encoder_iriscode_path, LOCAL_IRISCODE_DIR)
copy_all_files(encoder_iriscode_masks_path, LOCAL_IRISCODE_MASKS_DIR)

print(f"  → irisCode saved to        {LOCAL_IRISCODE_DIR}/")
print(f"  → irisCode_masks saved to  {LOCAL_IRISCODE_MASKS_DIR}/")

# =========================
# STEP 4: Prepare & run MATCHER
# =========================
print("\n[STEP 4] Preparing inputs for matcher...")

# If the matcher is using a different project folder than the encoder,
# make sure the matcher tree has the encoder outputs it needs.
if os.path.abspath(MATCHER_PATH) != os.path.abspath(ENCODER_PATH):
    matcher_iriscode_path = os.path.join(MATCHER_PATH, "data", "irisCode")
    matcher_iriscode_masks_path = os.path.join(MATCHER_PATH, "data", "irisCode_masks")
    ensure_dir(matcher_iriscode_path)
    ensure_dir(matcher_iriscode_masks_path)
    copy_all_files(encoder_iriscode_path, matcher_iriscode_path)
    copy_all_files(encoder_iriscode_masks_path, matcher_iriscode_masks_path)
    print("  → Copied encoder outputs into matcher tree.")

print("[STEP 4] Running matcher container...")
run_docker_once(
    name=MATCHER_CONTAINER_NAME,
    image=MATCHER_IMAGE,
    host_mount=MATCHER_PATH,
    workdir_in_container="/host",
    command=MATCHER_COMMAND
)
print("[STEP 4] Matching finished.")

# =========================
# STEP 5: Collect MATCHER outputs to local folders
# =========================
print("\n[STEP 5] Collecting matcher outputs...")

matcher_output_path = os.path.join(MATCHER_PATH, MATCHER_OUTPUT_FOLDER)
copy_all_files(matcher_output_path, LOCAL_DISTANCES_DIR)

print(f"  → distances/results saved to {LOCAL_DISTANCES_DIR}/")

print("\n✅ Pipeline complete.")

