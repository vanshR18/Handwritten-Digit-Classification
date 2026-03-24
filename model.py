

import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_resnet(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 adapted for MNIST.
    Changes from standard ResNet-18:
      - First conv: kernel 7→3, stride 2→1 (28×28 is too small for aggressive downsampling)
      - MaxPool after first conv removed (Identity)
      - Input channels: 3 → 1 (grayscale)
      - Final FC: 1000 → num_classes with Dropout
    """
    model = models.resnet18(weights=None)
    model.conv1   = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool  = nn.Identity()
    model.fc       = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model


def load_model(weights_path: str):
    """
    Load the ResNet-18 model from an inference bundle (.pth).
    Returns (model, checkpoint_dict).
    """
    checkpoint  = torch.load(weights_path, map_location=DEVICE)
    num_classes = checkpoint["num_classes"]

    model = build_resnet(num_classes).to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, checkpoint
