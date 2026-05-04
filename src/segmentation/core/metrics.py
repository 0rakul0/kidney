import torch


def _prepare_prediction(pred, threshold=0.5, from_logits=True):

    if from_logits:
        pred = torch.sigmoid(pred)

    return (pred > threshold).float()


def dice_score(pred, target, threshold=0.5, from_logits=True):

    pred = _prepare_prediction(pred, threshold=threshold, from_logits=from_logits)

    intersection = (pred * target).sum()
    dice = (2 * intersection) / (
        pred.sum() + target.sum() + 1e-8
    )

    return dice.item()


def iou_score(pred, target, threshold=0.5, from_logits=True):

    pred = _prepare_prediction(pred, threshold=threshold, from_logits=from_logits)

    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection

    return (intersection / (union + 1e-8)).item()

