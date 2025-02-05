import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from torch import nn
from torch.nn import functional as F
from tqdm.autonotebook import tqdm
from transformers import AutoImageProcessor, ResNetForImageClassification

from src.adversarial_noise.evaluation import plot_adversarial_results
from src.adversarial_noise.fgsm import (
    iterative_fast_gradient_sign_target,
    tensor_to_image,
)
from src.adversarial_noise.utils import get_device

# Create test_data directory if it doesn't exist
data_dir = "test_data"
os.makedirs("test_data", exist_ok=True)

dataset = load_dataset("huggingface/cats-image", trust_remote_code=False)
image_processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
image = dataset["test"]["image"][0]

# Save the image to test_data folder
image_path = os.path.join(data_dir, "cat_image.jpg")
image.save(image_path)

inputs = image_processor(image, return_tensors="pt")
