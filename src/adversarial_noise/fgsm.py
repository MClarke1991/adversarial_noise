import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import ResNetForImageClassification


def iterative_fast_gradient_sign_target(
    model: ResNetForImageClassification,
    imgs: torch.Tensor,
    labels: torch.Tensor,
    target_label: str,
    label_names: list,
    epsilon: float = 0.02,
    num_iter: int = 10,
    alpha: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target_label not in label_names:
        raise ValueError(f"Target label {target_label} not found in label_names")

    target_label_idx = torch.tensor(model.config.label2id[target_label])
    target_label_idx = target_label_idx.to(imgs.device)
    target_labels = torch.full(
        (imgs.size(0),), target_label_idx, dtype=torch.long, device=imgs.device
    )
    print(target_labels)

    adv_imgs = imgs.clone()

    for _ in range(num_iter):
        adv_imgs = adv_imgs.detach().requires_grad_()
        outputs = model(adv_imgs)
        logits = outputs.logits
        preds = F.log_softmax(logits, dim=-1)
        loss = -torch.nn.CrossEntropyLoss()(preds, target_labels)
        loss.sum().backward()

        noise_grad = torch.sign(adv_imgs.grad)
        adv_imgs = adv_imgs + alpha * noise_grad

    noise_grad = adv_imgs - imgs
    predicted_label = logits.argmax(-1).item()
    print(model.config.id2label[predicted_label])
    return adv_imgs.detach(), noise_grad.detach()

def tensor_to_image(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalized tensor to a PIL Image."""
    # Move to CPU and convert to numpy
    img_array = tensor.cpu().detach().numpy()

    # Reshape if needed (remove batch dimension if present)
    if len(img_array.shape) == 4:
        img_array = img_array[0]

    # Denormalize
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