from pathlib import Path
from typing import Any

import pytest
import torch
from transformers import AutoImageProcessor, ResNetForImageClassification

from adversarial_noise.fgsm import iterative_fast_gradient_sign_target
from adversarial_noise.utils import load_and_preprocess_image


@pytest.fixture
def model() -> ResNetForImageClassification:
    return ResNetForImageClassification.from_pretrained("microsoft/resnet-34")


@pytest.fixture
def image_processor() -> Any:
    return AutoImageProcessor.from_pretrained("microsoft/resnet-34")


@pytest.fixture
def sample_image(image_processor: Any) -> torch.Tensor:
    # Using a sample image from the test assets
    image_path = Path(__file__).parent / "assets" / "cat.jpg"
    return load_and_preprocess_image(image_path, image_processor)


def test_iterative_fgsm_target(
    model: ResNetForImageClassification,
    sample_image: torch.Tensor,
) -> None:
    # Setup
    label_names = list(model.config.label2id.keys())
    target_label = "dog"  # Trying to make the cat look like a dog
    
    # Run attack
    adv_image, noise = iterative_fast_gradient_sign_target(
        model=model,
        image=sample_image,
        target_label=target_label,
        label_names=label_names,
        num_iter=5,
        alpha=0.01,
        verbose=True,
    )
    
    # Basic assertions
    assert isinstance(adv_image, torch.Tensor)
    assert isinstance(noise, torch.Tensor)
    assert adv_image.shape == sample_image.shape
    assert noise.shape == sample_image.shape
    
    # Check if the adversarial image is different from the original
    assert not torch.allclose(adv_image, sample_image)
    
    # Verify the noise is non-zero
    assert torch.any(noise != 0)
    
    # Check model prediction on adversarial image
    with torch.no_grad():
        adv_logits = model(adv_image).logits
        adv_pred = adv_logits.argmax(-1).item()
        adv_label = model.config.id2label[adv_pred]
        
        # Get original prediction for comparison
        orig_logits = model(sample_image).logits
        orig_pred = orig_logits.argmax(-1).item()
        orig_label = model.config.id2label[orig_pred]
        
        # Verify predictions changed
        assert orig_label != adv_label, "Attack should change the model's prediction"


def test_iterative_fgsm_target_invalid_label(
    model: ResNetForImageClassification,
    sample_image: torch.Tensor,
) -> None:
    label_names = list(model.config.label2id.keys())
    
    with pytest.raises(ValueError):
        iterative_fast_gradient_sign_target(
            model=model,
            image=sample_image,
            target_label="nonexistent_label_xyz",
            label_names=label_names,
            num_iter=5,
            alpha=0.01,
        )