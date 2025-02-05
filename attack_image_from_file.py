import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

from adversarial_noise.fgsm import iterative_fast_gradient_sign_target, tensor_to_image
from adversarial_noise.utils import get_device


def load_and_preprocess_image(
    image_path: str | Path,
    image_processor: AutoImageProcessor,
) -> torch.Tensor:
    """Load and preprocess an image for the model."""
    image = Image.open(image_path).convert("RGB")
    inputs = image_processor(image, return_tensors="pt", use_fast=False)
    return inputs.pixel_values


def attack_image(
    image_path: str | Path,
    target_label: str,
    output_path: Optional[str | Path] = None,
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
        output_path = input_path.parent / f"{input_path.stem}_target_{target_label}_achieved_{final_label}{input_path.suffix}"
    
    adv_image_pil.save(output_path)
    
    if verbose:
        print(f"Saved adversarial image to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate adversarial image from file")
    parser.add_argument("image_path", type=str, help="Path to input image")
    parser.add_argument("target_label", type=str, help="Target class label")
    parser.add_argument("--output", type=str, help="Path to save output image", default=None)
    parser.add_argument("--alpha", type=float, default=0.02, help="Step size for attack")
    parser.add_argument("--num-iter", type=int, default=10, help="Number of iterations")
    parser.add_argument("--model", type=str, default="microsoft/resnet-34", help="Model name")
    parser.add_argument("--verbose", default=True, action="store_true", help="Print additional information")
    
    args = parser.parse_args()
    
    attack_image(
        image_path=args.image_path,
        target_label=args.target_label,
        output_path=args.output,
        alpha=args.alpha,
        num_iter=args.num_iter,
        model_name=args.model,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()