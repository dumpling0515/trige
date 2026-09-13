import torch
import torchxrayvision as xrv


MODEL_NAME = "densenet121-res224-all"


def load_model():
    model = xrv.models.DenseNet(weights=MODEL_NAME)
    model.eval()
    return model


MODEL = load_model()


def predict_xray(image_path):
    # TorchXRayVision handles the required intensity normalization.
    img = xrv.utils.load_image(str(image_path))

    # Standard preprocessing expected by this model.
    img = xrv.datasets.XRayCenterCrop()(img)
    img = xrv.datasets.XRayResizer(224)(img)

    # Add batch dimension:
    # [1, 224, 224] -> [1, 1, 224, 224]
    img_tensor = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        output = MODEL(img_tensor)[0]

    pneumonia_index = MODEL.pathologies.index("Pneumonia")
    pneumonia_score = float(output[pneumonia_index].item())

    return pneumonia_score