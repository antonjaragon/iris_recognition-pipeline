from modules.irisRecognition import irisRecognition
from modules.utils import get_cfg
import argparse
import glob
from PIL import Image
import os
import cv2
import csv
import numpy as np

# -------- Helpers -------------------------------------------------------------

def imread_gray_pil(path):
    # Keep single-channel PIL image ("L")
    return Image.fromarray(np.array(Image.open(path).convert("RGB"))[:, :, 0], "L")

def to_numpy_gray(im_pil):
    # PIL "L" -> uint8 numpy (H,W)
    return np.array(im_pil, dtype=np.uint8)

def resize_like(im_pil, target_size=(640, 480)):
    # target_size = (W, H)
    return im_pil.resize(target_size, resample=Image.BILINEAR)

def resize_mask_like(mask_np, target_size=(640, 480)):
    # nearest-neighbor to preserve labels
    mask_pil = Image.fromarray(mask_np)
    mask_resized = mask_pil.resize(target_size, resample=Image.NEAREST)
    return np.array(mask_resized, dtype=np.uint8)

def min_enclosing_circle_from_mask(mask_bin):
    """
    mask_bin: np.uint8, 0/255. Returns (x, y, r) in pixel coords (float).
    Uses outermost contour; falls back to equivalent radius if needed.
    """
    # OpenCV wants 0/255; ensure binary:
    mask_bin = (mask_bin > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        # Empty -> invalid
        return None
    # Choose the largest contour by area
    cnt = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(cnt)
    # Safety fallback: if contour degenerate, use moments/area
    if r <= 0:
        M = cv2.moments(cnt)
        if M["m00"] > 1e-6:
            cx = M["m10"] / M["m00"]
            cy = M["m01"] / M["m00"]
            area = cv2.contourArea(cnt)
            r_eq = np.sqrt(area / np.pi)
            return (cx, cy, r_eq)
        else:
            return None
    return (float(x), float(y), float(r))

def xy_radius_from_ring_mask(mask_ring):
    """
    If you have a single binary mask of the whole iris ring (pupil hole + iris),
    we approximate:
      - iris_xyr from the outer boundary,
      - pupil_xyr from the inner hole (largest hole).
    """
    mask = (mask_ring > 0).astype(np.uint8)
    # Outer boundary from mask itself
    iris = min_enclosing_circle_from_mask(mask * 255)

    # Infer the pupil hole as the largest connected component of ~mask inside iris bbox
    # More robust: find contours on the inverted mask & choose one inside iris circle.
    inv = (1 - mask).astype(np.uint8) * 255
    contours, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    if iris is not None:
        ix, iy, ir = iris
        for c in contours:
            area = cv2.contourArea(c)
            if area < 5:
                continue
            (x, y), r = cv2.minEnclosingCircle(c)
            # consider only holes well inside the iris
            if (x - ix) ** 2 + (y - iy) ** 2 < (ir * 0.9) ** 2:
                candidates.append((area, (float(x), float(y), float(r))))
    if candidates:
        candidates.sort(reverse=True)  # largest area first
        pupil = candidates[0][1]
    else:
        pupil = None

    return pupil, iris

def load_mask_pair(base_name, masks_dir):
    """
    Your setup: one mask per image with the same base name in data/masks/.
    Examples:
        image: data/images/subject01.tiff
        mask:  data/masks/subject01.png   (or .tif/.tiff/.jpg, etc.)
    We treat this single mask as a ring mask.
    """
    import glob, os
    from PIL import Image
    import numpy as np

    exts = ["png", "bmp", "jpg", "jpeg", "tif", "tiff", "gif"]

    # search for ANY extension with same base name (case-insensitive on ext)
    candidates = []
    for ext in exts:
        candidates.extend(glob.glob(os.path.join(masks_dir, f"{base_name}.{ext}")))
        candidates.extend(glob.glob(os.path.join(masks_dir, f"{base_name}.{ext.upper()}")))
    # also allow “anything with same base.*” as a last resort
    if not candidates:
        candidates = glob.glob(os.path.join(masks_dir, f"{base_name}.*"))

    if not candidates:
        # No mask found
        return None, None, None

    # If multiple matches, take the first (you can sort or pick by preferred ext)
    ring_path = sorted(candidates)[0]
    # print which mask we picked (helps debugging)
    print(f"[INFO] Using mask: {os.path.basename(ring_path)}")

    # Read & binarize
    m = np.array(Image.open(ring_path).convert("L"))
    # If your mask is already 0/255, this keeps it; if grayscale, threshold at 128:
    m = (m > 127).astype(np.uint8) * 255

    # We return it as a ring mask (single-mask workflow)
    return None, None, m


# -------- Main ---------------------------------------------------------------
import os, glob, csv, numpy as np
from itertools import combinations

def main(cfg, images_dir="./data/images", masks_dir="./data/masks", out_dir="./templates"):
    irisRec = irisRecognition(cfg)

    extensions = {"bmp", "png", "gif", "jpg", "jpeg", "tiff", "tif"}

    # 0) Gather all image paths recursively + their relative subfolder
    img_entries = []  # (abs_path, rel_dir, base)
    for root, _, files in os.walk(images_dir):
        for fname in files:
            ext = fname.rsplit(".", 1)[-1].lower()
            if ext in extensions:
                abs_path = os.path.join(root, fname)
                rel_dir = os.path.relpath(root, images_dir)  # "" or "." for top-level
                if rel_dir in (".",):
                    rel_dir = ""
                base = os.path.splitext(fname)[0]
                img_entries.append((abs_path, rel_dir, base))

    # Sort deterministically by (rel_dir, base)
    img_entries.sort(key=lambda x: (x[1], x[2]))

    # 1) Ensure roots exist
    os.makedirs(out_dir, exist_ok=True)
    outputs_root = "data/outputs"
    os.makedirs(outputs_root, exist_ok=True)

    # 2) Process images -> vectors (save templates in mirrored structure)
    vectors_list = []   # (rel_path_str, vector)
    for img_path, rel_dir, base in img_entries:
        rel_path_str = os.path.join(rel_dir, base) if rel_dir else base
        print(rel_path_str)

        # Load image
        im = imread_gray_pil(img_path)

        # Load masks from mirrored masks subdir
        masks_subdir = os.path.join(masks_dir, rel_dir) if rel_dir else masks_dir
        pupil_mask, iris_mask, ring_mask = load_mask_pair(base, masks_subdir)

        if pupil_mask is None and iris_mask is None and ring_mask is None:
            print(f"[WARN] No masks found for {rel_path_str}. Skipping.")
            continue

        # Fix image (and use its size for mask resizing)
        im_fixed = irisRec.fix_image(im)  # 640x480, preserves 4:3
        W, H = im_fixed.size

        # Resize masks identically
        if pupil_mask is not None:
            pupil_mask = resize_mask_like(pupil_mask, target_size=(W, H))
        if iris_mask is not None:
            iris_mask = resize_mask_like(iris_mask, target_size=(W, H))
        if ring_mask is not None:
            ring_mask = resize_mask_like(ring_mask, target_size=(W, H))

        # Circle params from masks (with ring fallback)
        if ring_mask is not None and (pupil_mask is None or iris_mask is None):
            pupil_xyr, iris_xyr = xy_radius_from_ring_mask(ring_mask)
        else:
            pupil_xyr = min_enclosing_circle_from_mask(pupil_mask) if pupil_mask is not None else None
            iris_xyr  = min_enclosing_circle_from_mask(iris_mask)  if iris_mask  is not None else None

        # Fallback to built-in approximator if needed
        if pupil_xyr is None or iris_xyr is None:
            pupil_xyr_fallback, iris_xyr_fallback = irisRec.circApprox(im_fixed)
            if pupil_xyr is None: pupil_xyr = pupil_xyr_fallback
            if iris_xyr  is None: iris_xyr  = iris_xyr_fallback

        # Normalize & extract vector
        im_polar = irisRec.cartToPol_torch(im_fixed, pupil_xyr, iris_xyr)
        vector = irisRec.extractVector(im_polar)

        # Save template in mirrored templates subdir
        tmpl_subdir = os.path.join(out_dir, rel_dir) if rel_dir else out_dir
        os.makedirs(tmpl_subdir, exist_ok=True)
        np.savez_compressed(os.path.join(tmpl_subdir, f"{base}_tmpl.npz"), vector)

        vectors_list.append((rel_path_str, vector))

    # 3) Global ALL-vs-ALL matching across every image gathered
    results = []
    for (fn1, v1), (fn2, v2) in combinations(vectors_list, 2):
        score = irisRec.matchVectors(v1, v2)
        print(f"{fn1} <-> {fn2} : {score:.3f}")
        results.append((fn1, fn2, score))

    # 4) Write a single GLOBAL CSV (paths are relative to images_dir root)
    out_path = os.path.join(outputs_root, "matching_results.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        # writer.writerow(["Image1", "Image2", "Score"])
        writer.writerows(results)

    print(f"[INFO] Global matching results saved to {out_path}")
    return None


# --------- CLI ---------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_path", type=str, default="cfg_baseline.yaml", help="path of the configuration file")
    parser.add_argument("--images_dir", type=str, default="./data/input_images", help="directory with input images")
    parser.add_argument("--masks_dir", type=str, default="./data/masks", help="directory with masks")
    parser.add_argument("--out_dir",   type=str, default="./templates", help="where to write templates")
    args = parser.parse_args()
    main(get_cfg(args.cfg_path), images_dir=args.images_dir, masks_dir=args.masks_dir, out_dir=args.out_dir)
