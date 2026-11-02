import pytest
import torch

from cxrtrust.gradcam import GradCAM
from cxrtrust.losses import AsymmetricLoss, MaskedBCE
from cxrtrust.models import CXRClassifier


@pytest.mark.parametrize("backbone", ["densenet121", "resnet50", "efficientnet_b0", "convnext_tiny"])
def test_backbones_output_shape(backbone):
    model = CXRClassifier(backbone, num_classes=5, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5)


def test_masked_bce_ignores_masked_labels():
    logits = torch.tensor([[5.0, -5.0]])
    target = torch.tensor([[1.0, 1.0]])
    full = MaskedBCE()(logits, target)
    masked = MaskedBCE()(logits, target, torch.tensor([[1.0, 0.0]]))
    assert masked < full
    assert masked == pytest.approx(torch.nn.functional.softplus(torch.tensor(-5.0)).item(), rel=1e-4)


def test_asl_reduces_to_bce_without_focusing():
    logits = torch.randn(8, 4)
    target = torch.randint(0, 2, (8, 4)).float()
    asl = AsymmetricLoss(gamma_pos=0, gamma_neg=0, clip=0)(logits, target)
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert asl.item() == pytest.approx(bce.item(), rel=1e-4)


@pytest.mark.parametrize("plus_plus", [False, True])
def test_gradcam_shape_and_range(plus_plus):
    torch.manual_seed(0)
    model = CXRClassifier("resnet50", num_classes=3, pretrained=False)
    with GradCAM(model, model.target_layer, plus_plus=plus_plus) as cam:
        heat = cam(torch.randn(2, 3, 64, 64), class_idx=1)
    assert heat.shape == (2, 64, 64)
    assert heat.min() >= 0 and heat.max() <= 1 + 1e-6
