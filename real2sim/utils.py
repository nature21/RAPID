import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import matplotlib.patches as patches
from skimage.color import lab2rgb, rgb2lab
from sklearn.cluster import KMeans
import numpy as np
from io import BytesIO
import base64
from PIL import Image
import copy


###############################################
# AFFORDANCE DETECTION
###############################################
AFFORDANCE_COLORS = {
    "tool_grasp_part": (255, 0, 0),      # red
    "tool_function_part": (0, 255, 0),       # green
    "target_function_part": (0, 0, 255)        # blue
}

def crop_image(image, bbox, offset_ratio=0.1):
    x1, y1, x2, y2 = bbox

    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    w_extend = int(w * (1 + offset_ratio))
    h_extend = int(h * (1 + offset_ratio))
    side = max(1, max(w_extend, h_extend))  # prevent side = 0

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_x1 = int(cx - side / 2)
    new_y1 = int(cy - side / 2)

    new_x1 = max(0, min(new_x1, image.width - 1))
    new_y1 = max(0, min(new_y1, image.height - 1))

    new_x2 = min(image.width,  new_x1 + side)
    new_y2 = min(image.height, new_y1 + side)

    if new_x2 <= new_x1:
        new_x2 = min(image.width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(image.height, new_y1 + 1)

    crop_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    cropped = image.crop(crop_box)

    return cropped, crop_box

def crop_image_and_apply_mask(image, bbox, mask, offset_ratio=0.1):
    """
    image: PIL.Image (RGB)
    mask:  numpy.ndarray, uint8, values {0,1}, shape (H,W)
    bbox:  (x1, y1, x2, y2)
    """

    # ---------- 1. Apply mask ----------
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)

    # Convert {0,1} -> {0,255}
    mask_255 = (mask * 255)

    # Convert to PIL mask
    mask_pil = Image.fromarray(mask_255, mode="L")

    # White background
    white_bg = Image.new("RGB", image.size, (255, 255, 255))

    # Composite image: mask==1 keeps color, else white
    masked_image = Image.composite(image, white_bg, mask_pil)

    if bbox is None:
        return masked_image, None

    x1, y1, x2, y2 = bbox
    x1, x2 = sorted([x1, x2])
    y1, y2 = sorted([y1, y2])

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)

    w_extend = int(w * (1 + offset_ratio))
    h_extend = int(h * (1 + offset_ratio))
    side = max(1, max(w_extend, h_extend))  # prevent side = 0

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    new_x1 = int(cx - side / 2)
    new_y1 = int(cy - side / 2)

    new_x1 = max(0, min(new_x1, image.width - 1))
    new_y1 = max(0, min(new_y1, image.height - 1))

    new_x2 = min(image.width,  new_x1 + side)
    new_y2 = min(image.height, new_y1 + side)

    if new_x2 <= new_x1:
        new_x2 = min(image.width, new_x1 + 1)
    if new_y2 <= new_y1:
        new_y2 = min(image.height, new_y1 + 1)

    crop_box = (int(new_x1), int(new_y1), int(new_x2), int(new_y2))
    cropped = masked_image.crop(crop_box)

    return cropped, crop_box

def apply_background_mask(image, mask, background_image_dir):
    """
    Apply a binary mask to an image, setting masked-out regions to a background image.

    Args:
        image (PIL.Image): Input image.
        mask (np.ndarray): Binary mask of shape (H, W), with 1 for foreground and 0 for background.
        background_image_dir (str): Path to the background image file.
    """
    image_np = np.array(image)
    mask_3ch = np.stack([mask]*3, axis=-1)  # (H, W, 3)

    background_image = Image.open(background_image_dir).convert("RGB")
    background_image = background_image.resize((image.width, image.height))
    background_np = np.array(background_image)

    output_np = image_np * mask_3ch + background_np * (1 - mask_3ch)
    output_image = Image.fromarray(output_np.astype(np.uint8))

    return output_image

def project_mask_to_original(aff_mask, crop_bbox, original_size):
    W, H = original_size
    full_mask = np.zeros((H, W), dtype=np.uint8)

    x1, y1, x2, y2 = crop_bbox
    crop_h = y2 - y1
    crop_w = x2 - x1

    # Resize mask to match crop size
    if aff_mask.shape != (crop_h, crop_w):
        aff_mask = Image.fromarray(aff_mask).resize((crop_w, crop_h), Image.NEAREST)
        aff_mask = np.array(aff_mask)

    full_mask[y1:y2, x1:x2] = aff_mask
    return full_mask

def apply_affordance_color(original_img, mask, aff_type):
    img = copy.deepcopy(original_img)
    r, g, b = AFFORDANCE_COLORS[aff_type]
    img[mask > 0] = img[mask > 0] + [r, g, b]
    return img

def count_affordance_values(featuremap, channel=0):
    # featurcemap:  (H, W,C), channel 0: grasp, channel 1: action, channel 2: interaction
    affordance_feature = featuremap[:,:, channel]  # (H,W)
    counts = np.sum(affordance_feature > 0)
    return counts


###############################################
# OBJECT detection UTILS
###############################################
def convert_pil_image_to_base64(image: Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def decode_base64(base64_image):
    image_data = base64.b64decode(base64_image)
    image_data = BytesIO(image_data)
    image = Image.open(image_data).convert("RGB")
    return image

def xyxy_to_xywh(box):
    """
    Convert [x1, y1, x2, y2] to [x, y, w, h].
    """
    if box is None:
        return None
    x1, y1, x2, y2 = box
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2 - x1, y2 - y1]

###############################################
# visualization
###############################################

def generate_colors(n_colors=256, n_samples=5000):
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb

COLORS = generate_colors(n_colors=128, n_samples=5000)

def plot_mask(mask, color="r", ax=None):
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * 0.5
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)

def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (x, y),
        w,
        h,
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            x,
            y - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )

def plot_best(image, mask, bbox, score, save=False, save_path="seg.png", show=True):
    plt.figure(figsize=(12, 8))
    plt.imshow(image)

    if mask is not None:
        plot_mask(mask, color="#FF4500")

    if bbox is not None:
        if score is None:
            plot_bbox(image.size[1], image.size[0], bbox, text=None,
                      box_format="XYXY", color="b", relative_coords=False)
        else:
            plot_bbox(image.size[1], image.size[0], bbox, text=f"Score: {score:.2f}",
                      box_format="XYXY", color="b", relative_coords=False)

    plt.axis("off")  # hide axis

    if save:
        plt.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()
    plt.close()

def plot_all(img, masks, bboxs, scores):
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    nb_objects = len(scores)
    print(f"found {nb_objects} object(s)")
    for i in range(nb_objects):
        color = COLORS[i % len(COLORS)]
        plot_mask(masks[i], color=color)
        w, h = img.size
        prob = scores[i]
        plot_bbox(
            h,
            w,
            bboxs[i],
            text=f"(id={i}, {prob=:.2f})",
            box_format="XYXY",
            color=color,
            relative_coords=False,
        )


###############################################
# Metric computation
###############################################
def compute_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0.0
    iou = intersection / union
    return iou