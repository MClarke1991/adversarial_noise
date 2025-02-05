import matplotlib.pyplot as plt
import plotly.graph_objects as go
import torch
from plotly.subplots import make_subplots
from transformers import ResNetForImageClassification

from adversarial_noise.fgsm import tensor_to_image


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
        rows=2, cols=3,
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
        row=2, col=1
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