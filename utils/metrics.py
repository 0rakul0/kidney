import torch


def dice_score(pred, target, threshold=0.5, from_logits=True):

    if from_logits:
        pred = torch.sigmoid(pred)

    pred = (pred > threshold).float()

    intersection = (pred * target).sum()

    dice = (2 * intersection) / (
        pred.sum() + target.sum() + 1e-8
    )

    return dice.item()
