import argparse
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification

from adversarial_noise.fgsm import (
    attack_image,
    iterative_fast_gradient_sign_target,
    tensor_to_image,
)
from adversarial_noise.utils import get_device, load_and_preprocess_image


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