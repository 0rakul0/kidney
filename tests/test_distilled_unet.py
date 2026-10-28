import numpy as np
import torch

from src.segmentation.experiments.train_distilled_unet import (
    distillation_loss,
    fuse_teacher_probabilities,
)


def test_teacher_fusion_is_probability_mean():
    first = np.array([[0.2, 0.8]], dtype=np.float32)
    second = np.array([[0.6, 0.4]], dtype=np.float32)

    result = fuse_teacher_probabilities(first, second)

    assert np.allclose(result, [[0.4, 0.6]])


def test_distillation_loss_is_finite_for_soft_targets():
    logits = torch.zeros((2, 1, 4, 4), dtype=torch.float32)
    teacher = torch.full((2, 4, 4), 0.7, dtype=torch.float32)

    result = distillation_loss(logits, teacher, temperature=2.0)

    assert torch.isfinite(result)
    assert result.item() > 0
