import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ResNetForImageClassification


def iterative_fast_gradient_sign_target(
    model: ResNetForImageClassification,
    image: torch.Tensor,
    target_label: str,
    label_names: list,
    num_iter: int = 10,
    alpha: float = 0.002,
    verbose: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative Fast Gradient Sign Method (FGSM) for targeted attacks.

    Args:
        model: Image classification model.
        image: Input images to attack.
        target_label: Target label for the attack.
        label_names: List of all label names recognized by the model.
        num_iter: Number of iterations for the attack.
        alpha: Step size for noise to add to the image.
        verbose: Whether to print target and predicted labels.

    Returns:
        Tuple of adversarial images and noise gradients.
    """
    
    if target_label not in label_names:
        raise ValueError(f"Target label {target_label} not found in label_names")

    target_label_idx = torch.tensor(model.config.label2id[target_label])
    target_label_idx = target_label_idx.to(image.device)
    target_labels = torch.full(
        (image.size(0),), target_label_idx, dtype=torch.long, device=image.device
    )

    adv_image = image.clone()

    for _ in range(num_iter):
        adv_image = adv_image.detach().requires_grad_()
        outputs = model(adv_image)
        logits = outputs.logits
        preds = F.log_softmax(logits, dim=-1)
        loss = -torch.nn.CrossEntropyLoss()(preds, target_labels)
        loss.sum().backward()

        noise_grad = torch.sign(adv_image.grad)
        adv_image = adv_image + alpha * noise_grad

    noise_grad = adv_image - image
    
    predicted_id = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_id]
    
    if predicted_label != target_label:
        print(f"Failed to target {target_label} with {predicted_label}. Increase alpha or num_iter.")
    
    if verbose:
        print(f"Target label: {target_label}, Achieved label: {predicted_label}")
    return adv_image.detach(), noise_grad.detach()

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor to a PIL Image."""
    # Move to CPU and convert to numpy
    img_array = tensor.cpu().detach().numpy()

    # Reshape if needed (remove batch dimension if present)
    if len(img_array.shape) == 4:
        img_array = img_array[0]

    # Denormalize
    # Note that these apparently hardcoded numbers are from https://pytorch.org/vision/0.9/models.html
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_array = std[:, None, None] * img_array + mean[:, None, None]

    # Clip values to valid range
    img_array = np.clip(img_array, 0, 1)

    # Convert to uint8
    img_array = (img_array * 255).astype(np.uint8)

    # Transpose from (C, H, W) to (H, W, C)
    img_array = np.transpose(img_array, (1, 2, 0))

    # Convert to PIL Image
    return Image.fromarray(img_array)