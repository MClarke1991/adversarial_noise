from difflib import get_close_matches
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

from adversarial_noise.utils import get_device, load_and_preprocess_image


def fuzzy_match_label(
    target: str, available_labels: list[str], cutoff: float = 0.6
) -> str:
    """
    Find the closest matching label from available labels using fuzzy matching.

    Args:
        target: The target label to match
        available_labels: List of valid labels to match against
        cutoff: Minimum similarity score (0-1) for matches

    Returns:
        The exact match if found, otherwise the closest matching label

    Raises:
        ValueError: If no matches found above cutoff threshold
    """
    # First try exact match
    if target.lower() in [label.lower() for label in available_labels]:
        return next(label for label in available_labels if label.lower() == target.lower())
    
    # Try substring matching first
    substring_matches = [
        label for label in available_labels 
        if target.lower() in label.lower() or label.lower() in target.lower()
    ]
    if substring_matches:
        closest_match = substring_matches[0]
        print(f"Found substring match: '{target}' → '{closest_match}'")
        return closest_match

    # Fall back to difflib matching
    matches = get_close_matches(target, available_labels, n=1, cutoff=cutoff)
    if not matches:
        raise ValueError(
            f"No similar labels found for '{target}'. Available labels: {', '.join(available_labels)}"
        )

    closest_match = matches[0]
    print(f"Using fuzzy match: '{target}' → '{closest_match}'")
    return closest_match


# def iterative_fast_gradient_sign_target(
#     model: ResNetForImageClassification,
#     image: torch.Tensor,
#     target_label: str,
#     label_names: list,
#     num_iter: int = 10,
#     alpha: float = 0.002,
#     verbose: bool = False,
# ) -> tuple[torch.Tensor, torch.Tensor]:
#     """
#     Iterative Fast Gradient Sign Method (FGSM) for targeted attacks.

#     Args:
#         model: Image classification model.
#         image: Input images to attack.
#         target_label: Target label for the attack.
#         label_names: List of all label names recognized by the model.
#         num_iter: Number of iterations for the attack.
#         alpha: Step size for noise to add to the image.
#         verbose: Whether to print target and predicted labels.

#     Returns:
#         Tuple of adversarial images and noise gradients.
#     """

#     target_label = fuzzy_match_label(target_label, label_names)

#     target_label_idx = torch.tensor(model.config.label2id[target_label])
#     target_label_idx = target_label_idx.to(image.device)
#     target_labels = torch.full(
#         (image.size(0),), fill_value=target_label_idx, dtype=torch.long, device=image.device
#     )

#     adv_image = image.clone()

#     for _ in range(num_iter):
#         adv_image = adv_image.detach().requires_grad_()
#         outputs = model(adv_image)
#         logits = outputs.logits
#         preds = F.log_softmax(logits, dim=-1)
#         loss = -torch.nn.CrossEntropyLoss()(preds, target_labels)
#         loss.sum().backward()

#         if adv_image.grad is None:
#             raise ValueError("Gradient is None")

#         noise_grad = torch.sign(adv_image.grad)
#         adv_image = adv_image + alpha * noise_grad

#     noise_grad = adv_image - image

#     predicted_id = logits.argmax(-1).item()
#     predicted_label = model.config.id2label[predicted_id]

#     if predicted_label != target_label:
#         print(
#             f"Failed to target {target_label} with {predicted_label}. Increase alpha or num_iter."
#         )

#     if verbose:
#         print(f"Target label: {target_label}, Achieved label: {predicted_label}")
#     return adv_image.detach(), noise_grad.detach()

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
    Handles normalization explicitly to ensure adversarial perturbations survive image conversion.

    Args:
        model: Image classification model.
        image: Input images to attack (assumed to be normalized).
        target_label: Target label for the attack.
        label_names: List of all label names recognized by the model.
        num_iter: Number of iterations for the attack.
        alpha: Step size for noise to add to the image.
        verbose: Whether to print target and predicted labels.

    Returns:
        Tuple of adversarial images and noise gradients.
    """
    # ImageNet normalization constants
    mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)

    target_label = fuzzy_match_label(target_label, label_names)
    target_label_idx = torch.tensor(model.config.label2id[target_label])
    target_label_idx = target_label_idx.to(image.device)
    target_labels = torch.full(
        (image.size(0),),
        fill_value=target_label_idx,
        dtype=torch.long,
        device=image.device,
    )

    # Denormalize the input image to 0-1 range
    denorm_image = image * std + mean
    adv_image = denorm_image.clone()

    for _ in range(num_iter):
        # Normalize the image before model input
        norm_adv_image = (adv_image - mean) / std
        norm_adv_image = norm_adv_image.detach().requires_grad_()

        outputs = model(norm_adv_image)
        logits = outputs.logits
        preds = F.log_softmax(logits, dim=-1)
        loss = -torch.nn.CrossEntropyLoss()(preds, target_labels)
        loss.sum().backward()

        if norm_adv_image.grad is None:
            raise ValueError("Gradient is None")

        # Scale the gradient back to denormalized space
        scaled_grad = norm_adv_image.grad * std
        noise_grad = torch.sign(scaled_grad)

        # Update in denormalized space
        adv_image = adv_image + alpha * noise_grad

        # Clip to valid image range
        adv_image = torch.clamp(adv_image, 0, 1)

    # Calculate noise in denormalized space
    noise_grad = adv_image - denorm_image

    # Final prediction using normalized image
    final_norm_image = (adv_image - mean) / std
    with torch.no_grad():
        final_logits = model(final_norm_image).logits
        predicted_id = final_logits.argmax(-1).item()
        predicted_label = model.config.id2label[predicted_id]

    if predicted_label != target_label:
        print(
            f"Failed to target {target_label} with {predicted_label}. Increase alpha or num_iter."
        )

    if verbose:
        print(f"Target label: {target_label}, Achieved label: {predicted_label}")

    # Return the adversarial image in normalized space to match input
    normalized_adv_image = (adv_image - mean) / std
    return normalized_adv_image.detach(), noise_grad.detach()


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


def attack_image(
    image_path: str | Path,
    target_label: str,
    output_path: str | Path | None = None,
    alpha: float = 0.002,
    num_iter: int = 10,
    model_name: str = "microsoft/resnet-34",
    verbose: bool = False,
) -> None:
    """Attack an image from a file and save the result."""
    # Setup device and model
    device = get_device()
    model = ResNetForImageClassification.from_pretrained(model_name).to(device)
    image_processor = AutoImageProcessor.from_pretrained(model_name)

    # Load and preprocess image
    image_tensor = load_and_preprocess_image(image_path, image_processor)
    image_tensor = image_tensor.to(device)

    # Get initial prediction
    with torch.no_grad():
        initial_logits = model(image_tensor).logits
        initial_pred = initial_logits.argmax(-1).item()
        initial_label = model.config.id2label[initial_pred]

    if verbose:
        print(f"Initial prediction: {initial_label}")

    # Get all possible labels
    label_names = list(model.config.label2id.keys())

    # Generate adversarial image
    adv_image, noise = iterative_fast_gradient_sign_target(
        model=model,
        image=image_tensor,
        target_label=target_label,
        label_names=label_names,
        num_iter=num_iter,
        alpha=alpha,
        verbose=verbose,
    )

    # Convert to PIL image and save
    adv_image_pil = tensor_to_image(adv_image)

    if output_path is None:
        # Get final prediction
        with torch.no_grad():
            final_logits = model(adv_image).logits
            final_pred = final_logits.argmax(-1).item()
            final_label = model.config.id2label[final_pred]

        # Create output filename based on input with target and achieved labels
        input_path = Path(image_path)
        output_path = (
            input_path.parent
            / f"{input_path.stem}_target_{target_label}_achieved_{final_label}{input_path.suffix}"
        )

    adv_image_pil.save(output_path)

    if verbose:
        print(f"Saved adversarial image to {output_path}")
