from threading import RLock

import torch
import torchxrayvision as xrv
import streamlit as st
from captum.attr import LayerGradCam, LayerAttribution


MODEL_NAME = "densenet121-res224-all"


@st.cache_resource
def load_model():
    model = xrv.models.DenseNet(weights=MODEL_NAME)
    model.eval()
    return model


@st.cache_resource
def load_model_lock():
    # Captum temporarily attaches hooks to the shared cached model.
    return RLock()


MODEL = load_model()
MODEL_LOCK = load_model_lock()


def inspect_attention_layer():
    """Inspect the loaded architecture rather than assume a named sublayer."""
    modules = dict(MODEL.named_modules())
    layer = modules.get("features")
    if layer is None or not any(
        isinstance(module, torch.nn.Conv2d) for module in layer.modules()
    ):
        raise RuntimeError(
            "Expected a convolutional MODEL.features stage. "
            "Inspect print(MODEL) before selecting a different Grad-CAM layer."
        )
    return "features", layer


ATTENTION_LAYER_NAME, ATTENTION_LAYER = inspect_attention_layer()


def predict_xray(image_path):
    """Return (pneumonia_score, pathology-score dict, 224x224 NumPy map).

    The Model Attention Map estimates regions with greater positive influence
    on the pneumonia-associated model output. It does not prove pneumonia
    exists in those regions. Coordinates refer to the center-cropped input.
    """
    # Preserve the original normalization, crop, resize, and batch creation.
    img = xrv.utils.load_image(str(image_path))
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    pneumonia_index = MODEL.pathologies.index("Pneumonia")

    with MODEL_LOCK:
        # Verify the selected feature stage produces spatial activations in
        # the real forward path before using it for Grad-CAM.
        feature_shapes = []

        def record_shape(module, inputs, output):
            if isinstance(output, torch.Tensor):
                feature_shapes.append(tuple(output.shape))

        handle = ATTENTION_LAYER.register_forward_hook(record_shape)
        try:
            with torch.no_grad():
                output = MODEL(img_tensor)[0]
        finally:
            handle.remove()

        if (
            len(feature_shapes) != 1
            or len(feature_shapes[0]) != 4
            or min(feature_shapes[0][-2:]) <= 1
        ):
            raise RuntimeError(
                f"Grad-CAM requires spatial feature maps; observed {feature_shapes}."
            )

        pneumonia_score = float(output[pneumonia_index].item())
        all_pathology_scores = {
            name: float(output[index].item())
            for index, name in enumerate(MODEL.pathologies)
        }

        # Separate attribution pass: use exactly the same model output,
        # including its existing output transformations. No weights change.
        with torch.enable_grad():
            attribution_input = img_tensor.detach().clone().requires_grad_(True)
            grad_cam = LayerGradCam(MODEL, ATTENTION_LAYER)
            attribution = grad_cam.attribute(
                attribution_input,
                target=pneumonia_index,
                relu_attributions=True,
            )
            upsampled = LayerAttribution.interpolate(
                attribution,
                (224, 224),
                interpolate_mode="bilinear",
            )

        heatmap = upsampled.detach()[0, 0].cpu()
        if not torch.isfinite(heatmap).all():
            raise RuntimeError("Grad-CAM returned non-finite attribution values.")

        low = heatmap.min()
        span = heatmap.max() - low
        if span.item() > 0:
            heatmap = ((heatmap - low) / span).clamp(0, 1)
        else:
            # Constant attribution contains no spatial contrast. Do not
            # manufacture highlighted regions or divide by zero.
            heatmap = torch.zeros_like(heatmap)

        attention_map = heatmap.numpy()

    return pneumonia_score, all_pathology_scores, attention_map
