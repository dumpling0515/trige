"""Display-only helpers; never modify inference inputs or model outputs."""

import numpy as np
from matplotlib import colormaps


def normalized_xray_to_rgb(normalized_xray):
    """Convert an XRV [-1024, 1024] image to a new H x W x 3 uint8 array."""
    image = np.array(normalized_xray, dtype=np.float32, copy=True)
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.ndim != 2 or not np.isfinite(image).all():
        raise ValueError("Expected a finite grayscale image with shape H x W or 1 x H x W.")
    grayscale = np.clip((image + 1024.0) / 2048.0, 0.0, 1.0)
    rgb = np.repeat(grayscale[..., None], 3, axis=2)
    return np.rint(rgb * 255.0).astype(np.uint8)


def create_attention_overlay(normalized_xray, attention_map, alpha=0.45):
    """Return an RGB uint8 overlay without changing either input.

    Both inputs must describe the same crop and pixel dimensions. Attention
    controls opacity: zero attribution leaves the original pixel unchanged;
    maximum attribution uses alpha opacity. No image rescaling is performed.
    """
    if not np.isfinite(alpha) or not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    original = normalized_xray_to_rgb(normalized_xray)
    attention = np.array(attention_map, dtype=np.float32, copy=True)
    if attention.shape != original.shape[:2]:
        raise ValueError("Attention map and radiograph must have matching spatial dimensions.")
    if not np.isfinite(attention).all():
        raise ValueError("Attention map must contain only finite values.")
    if np.any(attention < 0) or np.any(attention > 1):
        raise ValueError("Attention map must be normalized to the range 0–1.")

    heatmap_rgb = colormaps["inferno"](attention)[..., :3]
    opacity = (alpha * attention)[..., None]
    blended = (1.0 - opacity) * (original / 255.0) + opacity * heatmap_rgb
    return np.rint(np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def prepare_attention_views(image_path, attention_map):
    """Reproduce existing deterministic preprocessing for display alignment.

    This reloads a display copy only. It does not run inference or change the
    existing prediction function's return values.
    """
    import torchxrayvision as xrv

    image = xrv.utils.load_image(str(image_path))
    image = xrv.datasets.XRayCenterCrop()(image)
    image = xrv.datasets.XRayResizer(224)(image)
    return normalized_xray_to_rgb(image), create_attention_overlay(image, attention_map)


def rank_attention_regions(attention_map):
    """Return six dicts sorted by descending mean attention.

    Coordinates refer only to the image as displayed, never patient anatomy.
    Rows use [0:75], [75:150], [150:224]; columns use [0:112],
    [112:224]. Every pixel contributes once. Means account for unequal
    band heights. Exact ties retain top-to-bottom, image-left-first order
    for reproducibility; this ordering does not imply stronger attention.
    """
    attention = np.array(attention_map, dtype=np.float64, copy=True)
    if attention.shape != (224, 224):
        raise ValueError("Regional analysis requires a 224 × 224 attention map.")
    if not np.isfinite(attention).all():
        raise ValueError("Attention map must contain only finite values.")
    if np.any(attention < 0) or np.any(attention > 1):
        raise ValueError("Attention map must be normalized to the range 0–1.")

    ranked = []
    for band_name, start, stop in (
        ("Upper", 0, 75), ("Middle", 75, 150), ("Lower", 150, 224)
    ):
        for side, left, right in (("left", 0, 112), ("right", 112, 224)):
            ranked.append({
                "region": f"{band_name} image-{side} region",
                "mean_attention": float(attention[start:stop, left:right].mean()),
            })
    return sorted(ranked, key=lambda region: region["mean_attention"], reverse=True)


def attention_focus_labels(ranked_regions):
    """Format two focus labels without presenting arbitrary ties as winners."""
    labels = []
    for region in ranked_regions[:2]:
        tied = sum(
            np.isclose(region["mean_attention"], other["mean_attention"],
                       rtol=1e-6, atol=1e-8)
            for other in ranked_regions
        )
        if tied == len(ranked_regions):
            labels.append("No distinct regional focus")
        elif tied > 1:
            labels.append(region["region"] + " (tied)")
        else:
            labels.append(region["region"])
    return tuple(labels)
