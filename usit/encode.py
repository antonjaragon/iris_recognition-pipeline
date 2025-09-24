import subprocess
import cv2
import numpy as np
from pathlib import Path
from util.common import *
import logging

# Set up logging
logging.basicConfig(
    filename='errors.log',
    filemode='a',  # Append mode
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.ERROR
)

textures_list = []
iris_codes_list = []
titles_list = []

# Base paths (already mounted inside the container at /app)
masks_base = Path('./data/masks')
circles_base = Path('./data/circles')
input_base = Path('./data/input_images')
textures_base = Path('./data/textures')
textures_masks_base = Path('./data/texture_masks')
cnn_script = Path('./packages/cnnmasktomanuseg/cnnmasktomanuseg.py')
iris_code_base = Path('./data/irisCode')
iris_code_mask_base = Path('./data/irisCode_masks')
codes_base = Path('./data/codes')
codes_masks_base = Path('./data/code_masks')
complex_base = Path('./data/complex')


def run_cnn_script(mask_file, circle_dir, angleincrements=18):
    """Run the CNN script inside the container (now directly, since we're already inside)."""
    cmd = [
        'python', str(cnn_script),
        str(mask_file),
        str(circle_dir),
        str(angleincrements)
    ]
    print(f"Running CNN mask to manuseg for: {mask_file}")
    subprocess.run(cmd, check=True)


def main():
    # --------- Step 1: Build tools once ---------
    # print("Running make install clean...")
    # subprocess.run(["make", "-f", "Makefile_linux.mak", "install", "clean"], check=True)

    # --------- Step 2: Process masks and generate textures ---------
    for mask_file in masks_base.rglob('*.png'):
        relative_path = mask_file.relative_to(masks_base)
        stem = mask_file.stem

        circle_dir = circles_base / relative_path.parent
        input_tiff = input_base / relative_path.with_suffix('.tiff')
        input_tif = input_base / relative_path.with_suffix('.tif')
        inner_txt = circle_dir / f'{stem}.inner.txt'
        outer_txt = circle_dir / f'{stem}.outer.txt'
        output_texture = textures_base / relative_path.with_name(f"{stem}.bmp")
        output_texture_masks = textures_masks_base / relative_path.with_name(f"{stem}.bmp")
        output_code = codes_base / relative_path.with_name(f"{stem}.bmp")
        output_code_mask = codes_masks_base / relative_path.with_name(f"{stem}.bmp")

        complex_csv_path = complex_base / relative_path.parent / f"{stem}.csv"
        complex_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # Ensure dirs exist
        circle_dir.mkdir(parents=True, exist_ok=True)
        output_texture.parent.mkdir(parents=True, exist_ok=True)
        output_texture_masks.parent.mkdir(parents=True, exist_ok=True)
        output_code.parent.mkdir(parents=True, exist_ok=True)
        output_code_mask.parent.mkdir(parents=True, exist_ok=True)

        # Run CNN script
        try:
            run_cnn_script(mask_file, circle_dir, angleincrements=18)
        except subprocess.CalledProcessError as e:
            logging.error(f"CNN script failed for {mask_file}: {e}")
            continue

        # Pick input image
        input_image = input_tiff if input_tiff.exists() else input_tif
        if not input_image.exists():
            logging.error(f"No input image found for {mask_file}")
            continue

        # Run manuseg for textures
        cmd2 = [
            'manuseg',
            '-i', str(input_image),
            '-c', str(inner_txt), str(outer_txt),
            '', '',
            '-o', str(output_texture),
            '-q'
        ]

        cmd3 = [
            'manuseg',
            '-i', str(mask_file),
            '-c', str(inner_txt), str(outer_txt),
            '', '',
            '-o', str(output_texture_masks),
            '-q'
        ]

        cmd4 = [
            'lg',
            '-i', str(output_texture),
            '-m', str(output_texture_masks), str(output_code_mask),
            '-c', str(complex_csv_path),
            '-q',
            '-o', str(output_code)
        ]

        try:
            subprocess.run(cmd2, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"manuseg cmd2 failed for {input_image}: {e}")
            continue

        try:
            subprocess.run(cmd3, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"manuseg cmd3 failed for {input_image}: {e}")
            continue

        # Remove reflections from mask
        remove_reflections(
            texture_path=str(output_texture),
            mask_path=str(output_texture_masks),
            threshold=128,
            visualize=False
        )

        try:
            subprocess.run(cmd4, check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"lg cmd4 failed for {output_texture}: {e}")
            continue

        # --------- Step 3: Generate IrisCode ---------
        try:
            image = cv2.imread(str(output_code), cv2.IMREAD_GRAYSCALE)
            texture = cv2.imread(str(output_texture), cv2.IMREAD_GRAYSCALE)
            code_mask = cv2.imread(str(output_code_mask), cv2.IMREAD_GRAYSCALE)

            binary_code = calculate_usit_code(
                img=image,
                visualize=False,
                interleave_mode='interleave-horz',
                texture=texture,
                plot_grayscale=False
            )

            mask_binary_code = calculate_usit_code(
                img=code_mask,
                visualize=False,
                interleave_mode='interleave-horz',
                texture=texture,
                plot_grayscale=False
            )

            iris_code_dir = iris_code_base / relative_path.parent
            iris_code_dir.mkdir(parents=True, exist_ok=True)

            iris_code_image_path = iris_code_dir / f"{stem}.bmp"
            save_binary_code_as_image(binary_code, str(iris_code_image_path))
            print(f"Binary code saved as: {iris_code_image_path}")

            iris_code_mask_dir = iris_code_mask_base / relative_path.parent
            iris_code_mask_dir.mkdir(parents=True, exist_ok=True)

            iris_code_mask_image_path = iris_code_mask_dir / f"{stem}.bmp"
            save_binary_code_as_image(mask_binary_code, str(iris_code_mask_image_path))
            print(f"Mask code saved as: {iris_code_mask_image_path}")

            textures_list.append(output_texture)
            iris_codes_list.append(iris_code_image_path)
            titles_list.append(stem)

        except Exception as e:
            logging.error(f"Failed to generate/save iris code for {mask_file}: {e}")
            continue

    print("\n✅ All files processed inside container.")


if __name__ == "__main__":
    main()
