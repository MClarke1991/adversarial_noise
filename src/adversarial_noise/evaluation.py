import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from plotly.subplots import make_subplots
from tqdm.autonotebook import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification

from adversarial_noise.fgsm import iterative_fast_gradient_sign_target, tensor_to_image


def plot_adversarial_results(
    original_img: torch.Tensor,
    adv_img: torch.Tensor,
    noise_grad: torch.Tensor,
    model: ResNetForImageClassification,
    device: torch.device,
) -> None:
    """Plot adversarial attack results using plotly."""
    # Convert tensors to images
    orig_img = tensor_to_image(original_img)
    adv_img_pil = tensor_to_image(adv_img)
    noise_img = tensor_to_image(noise_grad)

    # Get model predictions
    with torch.no_grad():
        logits = model(adv_img.to(device)).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=10)
        labels = [model.config.id2label[idx.item()] for idx in top_indices]
        probs = [prob.item() * 100 for prob in top_probs]

    # Create subplot figure with 2 rows
    fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=("Original", "Adversarial", "Noise", "Top 10 Predictions"),
        row_heights=[0.5, 0.5],
        specs=[
            [{"type": "image"}, {"type": "image"}, {"type": "image"}],
            [{"type": "bar", "colspan": 3}, None, None],
        ],
    )

    # Add images to top row
    fig.add_trace(go.Image(z=orig_img), row=1, col=1)
    fig.add_trace(go.Image(z=adv_img_pil), row=1, col=2)
    fig.add_trace(go.Image(z=noise_img), row=1, col=3)

    # Add bar chart to bottom row
    fig.add_trace(
        go.Bar(
            x=probs,
            y=labels,
            orientation="h",
            text=[f"{p:.1f}%" for p in probs],
            textposition="outside",
        ),
        row=2,
        col=1,
    )

    # Update layout
    fig.update_layout(
        title_text="Adversarial Attack Results",
        showlegend=False,
        height=800,
        width=1200,
        bargap=0.2,
    )

    # Update yaxis properties for better bar chart display
    fig.update_yaxes(automargin=True, row=2, col=1)
    fig.show()


def evaluate_adversarial_attacks(
    model: ResNetForImageClassification,
    dataset: dict,
    image_processor: AutoImageProcessor,
    n_samples: int = 100,
    alpha: float = 0.002,
    num_iter: int = 10,
    device: str = "cuda",
    seed: int = 42,
) -> dict:
    """
    Evaluate adversarial attacks on multiple images with random targets.

    Args:
        model: The model to attack
        dataset: The dataset containing images
        image_processor: Processor to prepare images for the model
        n_samples: Number of attacks to attempt
        alpha: Step size for FGSM
        num_iter: Number of iterations for the attack
        device: Device to run the computations on

    Returns:
        dict: Results containing success rate and detailed statistics
    """
    # Get label names
    label_names = list(model.config.id2label.values())

    # Track results
    results = {
        "successful": 0,
        "failed": 0,
        "original_predictions": [],
        "target_labels": [],
        "final_predictions": [],
        "confidence_scores": [],
    }

    # Random indices for images

    random_indices = np.random.choice(len(dataset["train"]), n_samples)

    # Progress bar
    pbar = tqdm(random_indices, desc="Running adversarial attacks")

    for idx in pbar:
        # Get image and ensure it's in RGB format
        image = dataset["train"]["image"][idx]
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Process image
        inputs = image_processor(image, return_tensors="pt")
        image_tensor = inputs["pixel_values"].to(device)

        # Get original prediction
        with torch.no_grad():
            logits = model(image_tensor).logits
            original_pred = label_names[logits.argmax(-1).item()]

        # Select random target different from original prediction
        possible_targets = [l for l in label_names if l != original_pred]
        target_label = np.random.choice(possible_targets)

        # Attempt adversarial attack
        try:
            adv_imgs, _ = iterative_fast_gradient_sign_target(
                model=model,
                image=image_tensor,
                target_label=target_label,
                label_names=label_names,
                alpha=alpha,
                num_iter=num_iter,
            )

            # Check final prediction
            with torch.no_grad():
                logits = model(adv_imgs.to(device)).logits
                probs = F.softmax(logits, dim=-1)[0]
                final_pred = label_names[logits.argmax(-1).item()]
                confidence = probs.max().item()

            # Track results
            if final_pred == target_label:
                results["successful"] += 1
            else:
                results["failed"] += 1

            results["original_predictions"].append(original_pred)
            results["target_labels"].append(target_label)
            results["final_predictions"].append(final_pred)
            results["confidence_scores"].append(confidence)

        except Exception as e:
            print(f"Error on image {idx}: {str(e)}")
            results["failed"] += 1

    # Calculate success rate
    success_rate = (results["successful"] / n_samples) * 100
    print(f"\nSuccess rate: {success_rate:.2f}%")
    print(f"Failed attacks: {results['failed']}")

    return results