import matplotlib.pyplot as plt
import torch
from transformers import ResNetForImageClassification

from adversarial_noise.fgsm import tensor_to_image


def plot_adversarial_results(
    original_img: torch.Tensor,
    adv_img: torch.Tensor,
    noise_grad: torch.Tensor,
    model: ResNetForImageClassification,
    device: torch.device,
) -> None:
    """
    Plot adversarial attack results using matplotlib.

    Args:
        original_img: Original input image tensor
        adv_img: Adversarial image tensor
        noise_grad: Noise gradient tensor
        model: The model to get predictions from
        device: The device to run predictions on
    """

    # Create figure and axes with larger size
    fig = plt.figure(figsize=(25, 8))
    gs = fig.add_gridspec(1, 4, width_ratios=[1.2, 1.2, 1.2, 2])

    # Convert tensors to images
    orig_img = tensor_to_image(original_img)
    adv_img_pil = tensor_to_image(adv_img)
    noise_img = tensor_to_image(noise_grad)

    ax1 = fig.add_subplot(gs[0])
    ax1.imshow(orig_img)
    ax1.set_title("Original", fontsize=12, pad=10)
    ax1.axis("off")

    ax2 = fig.add_subplot(gs[1])
    ax2.imshow(adv_img_pil)
    ax2.set_title("Adversarial", fontsize=12, pad=10)
    ax2.axis("off")

    ax3 = fig.add_subplot(gs[2])
    ax3.imshow(noise_img)
    ax3.set_title("Noise", fontsize=12, pad=10)
    ax3.axis("off")

    # Get model predictions
    with torch.no_grad():
        logits = model(adv_img.to(device)).logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)[0]
        top_probs, top_indices = torch.topk(probabilities, k=10)

        # Get labels and probabilities
        labels = [model.config.id2label[idx.item()] for idx in top_indices]
        probs = [prob.item() * 100 for prob in top_probs]

    # Plot bar chart
    ax4 = fig.add_subplot(gs[3])
    bars = ax4.barh(range(len(probs)), probs)
    ax4.set_yticks(range(len(labels)))
    ax4.set_yticklabels(labels)
    ax4.set_xlabel("Probability (%)")
    ax4.set_title("Top 10 Predictions")

    # Add percentage labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax4.text(
            width + 1,
            bar.get_y() + bar.get_height() / 2,
            f"{probs[i]:.1f}%",
            va="center",
        )

    # Adjust layout
    plt.suptitle("Adversarial Attack Results", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()