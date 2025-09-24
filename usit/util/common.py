import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.gridspec import GridSpec



def calculate_usit_code(img, visualize=False,
                        interleave_mode='hstack', texture=None,
                        title=None, plot_grayscale=True, plot_binary=True):
    if img.ndim > 2:
        raise ValueError("Expected a 2D grayscale image.")
    if interleave_mode not in ('hstack', 'vstack', 'interleave-horz', 'interleave-vert'):
        raise ValueError(f"Invalid interleave_mode: {interleave_mode}")

    if texture is not None:
        tex_h, tex_w = texture.shape
    else:
        tex_h, tex_w = img.shape

    # In your C++ code, N = tex_h / M (number of vertical blocks after downscaling)
    # Here, let's just infer N from bits length
    img = img.astype(np.uint8)
    flat = img.flatten()
    bits = np.unpackbits(flat)

    # Total bits should be N * (2*w)
    # So N = total_bits / (2*w)
    total_bits = bits.size
    w = tex_w
    if total_bits % (2 * w) != 0:
        raise ValueError("Bits length not divisible by 2 * width")

    N = total_bits // (2 * w)

    # reshape as N rows, each row 2*w bits (real + imag side by side)
    bit_matrix = bits.reshape((N, 2 * w))

    # split horizontally
    real = bit_matrix[:, :w]
    imag = bit_matrix[:, w:]

    if interleave_mode == 'hstack':
        final = np.hstack((real, imag))

    elif interleave_mode == 'vstack':
        final = np.vstack((real, imag))

    elif interleave_mode == 'interleave-horz':
        final = np.empty((N, w * 2), dtype=real.dtype)
        final[:, 0::2] = real
        final[:, 1::2] = imag

    elif interleave_mode == 'interleave-vert':
        final = np.empty((N * 2, w), dtype=real.dtype)
        final[0::2] = real
        final[1::2] = imag


    if visualize:
        rows = []
        heights = []

        if texture is not None:
            rows.append("texture")
            heights.append(1)
        if plot_grayscale:
            rows.append("grayscale")
            heights.append(0.5)
        if plot_binary:
            rows.append("bit_vector")
            heights.append(0.5)
            rows.append("real_imag")
            heights.append(1)
            rows.append("final_matrix")
            heights.append(1)

        fig = plt.figure(figsize=(14, 3 * len(rows)), constrained_layout=False)
        gs = GridSpec(len(rows), 2, figure=fig,
                      height_ratios=heights,
                      hspace=0.4,
                      width_ratios=[1, 1])

        def apply_axis_format(ax, title, x_label, y_label, xtick_count, ytick_step, vmin, vmax):
            ax.set_title(title, pad=12)
            array = ax.images[0].get_array()
            ax.set_xticks(np.arange(0, array.shape[1], max(1, array.shape[1] // xtick_count)))
            rows_ax = array.shape[0]
            ytick_step = max(1, rows_ax // ytick_step)
            ax.set_yticks(np.arange(0, rows_ax, ytick_step))
            ax.set_yticklabels([f"{r}" for r in range(0, rows_ax, ytick_step)])
            ax.grid(False)

        plot_idx = 0
        for row_type in rows:
            if row_type == "texture":
                ax = fig.add_subplot(gs[plot_idx, :])
                tex = texture.reshape(1, -1) if texture.ndim == 1 else texture
                ax.imshow(tex, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto')
                apply_axis_format(ax, "Input Texture (64x512)", "Pixel Index", "Row", 8, 4, 0, 255)
                plot_idx += 1

            elif row_type == "grayscale":
                ax = fig.add_subplot(gs[plot_idx, :])
                img_display = img.reshape(1, -1) if img.ndim == 1 else img
                ax.imshow(img_display, cmap='gray', vmin=0, vmax=255, interpolation='nearest', aspect='auto')
                apply_axis_format(ax, "Input Grayscale Image", "Pixel Index", "Row", 8, 4, 0, 255)
                plot_idx += 1

            elif row_type == "bit_vector":
                ax = fig.add_subplot(gs[plot_idx, :])
                bit_strip = bits.reshape(1, -1)
                ax.imshow(bit_strip, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                apply_axis_format(ax, "Binary Vector (65536)", "Bit Index", "Row", 8, 1, 0, 1)
                plot_idx += 1

            elif row_type == "real_imag":
                ax_real = fig.add_subplot(gs[plot_idx, 0])
                ax_imag = fig.add_subplot(gs[plot_idx, 1])
                ax_real.imshow(real, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                apply_axis_format(ax_real, "Real Part (64x512)", "Pixel Index", "Row", 8, 4, 0, 1)
                ax_imag.imshow(imag, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                apply_axis_format(ax_imag, "Imaginary Part (64x512)", "Pixel Index", "Row", 8, 4, 0, 1)
                plot_idx += 1

            elif row_type == "final_matrix":
                ax = fig.add_subplot(gs[plot_idx, :])
                ax.imshow(final, cmap='gray', vmin=0, vmax=1, interpolation='nearest', aspect='auto')
                # apply_axis_format(ax, f"Final Binary Matrix ({interleave_mode})", "Pixel Index", "Row", 8, 4, 0, 1)
                apply_axis_format(ax, f"ReIm matrix (64x1024)", "Pixel Index", "Row", 8, 4, 0, 1)
                plot_idx += 1

        plt.suptitle(title if title else "USIT Binary Code Visualization", fontsize=16, y=0.99)
        plt.subplots_adjust(top=0.92)
        plt.show()

    return final


# Function to save binary code as an image
def save_binary_code_as_image(binary_code, save_path):
    """
    Saves the binary code (0s and 1s) as an image (black & white).
    
    Parameters:
        binary_code (numpy.ndarray): 2D binary matrix
        save_path (str): Path to save the output image
    """
    # Multiply by 255 to make it a visible image (0 = black, 1 = white)
    binary_image = (binary_code * 255).astype(np.uint8)
    cv2.imwrite(save_path, binary_image)



def remove_reflections(texture_path, mask_path, threshold=128, visualize=False, kernel_size=3):
    # === Load grayscale images ===
    texture = cv2.imread(texture_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if texture is None or mask is None:
        raise FileNotFoundError("Texture or mask file not found.")
    if texture.shape != mask.shape:
        raise ValueError("Texture and mask must be the same size.")

    # === Threshold logic ===
    bright_spots = (texture > threshold).astype(np.uint8) * 255

    # === Morphological opening to remove tiny regions ===
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    cleaned_spots = cv2.morphologyEx(bright_spots, cv2.MORPH_OPEN, kernel)

    # === Apply cleaned bright spots to mask ===
    modified_mask = mask.copy()
    modified_mask[cleaned_spots > 0] = 0
    binary_mask = np.where(modified_mask > 0, 255, 0).astype(np.uint8)

    # === Save updated mask ===
    cv2.imwrite(mask_path, binary_mask)

    # === Visualization ===
    if visualize:
        texture_rgb = np.stack([texture]*3, axis=-1)

        overlay = texture_rgb.copy()
        overlay[binary_mask == 0] = [255, 0, 0]

        fig, axs = plt.subplots(3, 1, figsize=(6, 12))
        axs[0].imshow(texture, cmap='gray')
        axs[0].set_title("Original Texture")
        axs[0].axis("off")

        axs[1].imshow(binary_mask, cmap='gray')
        axs[1].set_title("Modified Mask (Binarized)")
        axs[1].axis("off")

        axs[2].imshow(overlay)
        axs[2].set_title("Overlay (Red = Mask=0)")
        axs[2].axis("off")

        plt.tight_layout()
        plt.show()

    return binary_mask  



def plot_used_pixels_for_texture(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)  # or IMREAD_COLOR
    if image is None:
        raise FileNotFoundError("Could not load texture image.")

    # Convert to color if grayscale (to draw white dots clearly)
    if len(image.shape) == 2:
        texture_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        texture_vis = image.copy()

    # Load used coordinates from file
    points = np.loadtxt("used_texture_points.txt", dtype=int)

    # Draw each point as a white dot
    for x, y in points:
        cv2.circle(texture_vis, (x, y), radius=1, color=(255, 255, 255), thickness=-1)

    # Show the result
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(texture_vis, cv2.COLOR_BGR2RGB))
    plt.title(f"Image {image_path.split('/')[-1]}")
    plt.axis("off")
    plt.show()
