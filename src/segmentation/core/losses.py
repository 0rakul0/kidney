import torch
import torch.nn.functional as F


def _flatten_probs_and_target(logits, target):

    probs = torch.sigmoid(logits).reshape(-1)
    target = target.reshape(-1)

    return probs, target


def dice_loss(logits, target, smooth=1.0):

    probs, target = _flatten_probs_and_target(logits, target)

    intersection = (probs * target).sum()
    dice = (2 * intersection + smooth) / (
        probs.sum() + target.sum() + smooth
    )

    return 1 - dice


def tversky_loss(logits, target, alpha=0.5, beta=0.5, smooth=1.0):

    probs, target = _flatten_probs_and_target(logits, target)

    true_positive = (probs * target).sum()
    false_positive = (probs * (1 - target)).sum()
    false_negative = ((1 - probs) * target).sum()

    tversky = (true_positive + smooth) / (
        true_positive
        + alpha * false_positive
        + beta * false_negative
        + smooth
    )

    return 1 - tversky


def focal_loss(logits, target, alpha=0.25, gamma=2.0, reduction="mean"):

    target = target.float()
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")

    probs = torch.sigmoid(logits)
    pt = probs * target + (1 - probs) * (1 - target)
    focal_weight = (1 - pt).pow(gamma)

    if alpha is not None:
        alpha_factor = alpha * target + (1 - alpha) * (1 - target)
        focal_weight = focal_weight * alpha_factor

    loss = focal_weight * bce

    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss

    return loss.mean()


def focal_tversky_loss(
    logits,
    target,
    alpha=0.7,
    beta=0.3,
    gamma=1.33,
    smooth=1.0,
):

    tversky_value = 1 - tversky_loss(
        logits,
        target,
        alpha=alpha,
        beta=beta,
        smooth=smooth,
    )

    return (1 - tversky_value).pow(gamma)

