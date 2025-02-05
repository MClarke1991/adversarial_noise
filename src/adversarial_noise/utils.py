import torch


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def load_and_preprocess_image(
    image_path: str | Path,
    image_processor: AutoImageProcessor,
) -> torch.Tensor:
    """Load and preprocess an image for the model."""
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt", use_fast=False)
    return inputs.pixel_values