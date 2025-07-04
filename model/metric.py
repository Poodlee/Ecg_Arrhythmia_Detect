import torch

def accuracy(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> float:
    """
    Computes the overall multi-label accuracy (exact match ratio).

    Args:
        preds (torch.Tensor): Model predictions after sigmoid, shape (B, C).
        labels (torch.Tensor): Ground truth binary labels, shape (B, C).
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Mean accuracy across all elements (macro averaged).
    """
    preds = (preds > threshold).float()
    correct = (preds == labels).float()
    return correct.mean().item()


def sensitivity(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> float:
    """
    Computes macro-averaged sensitivity (recall) across all classes.

    Args:
        preds (torch.Tensor): Model predictions after sigmoid, shape (B, C).
        labels (torch.Tensor): Ground truth binary labels, shape (B, C).
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Macro-averaged sensitivity.
    """
    preds = (preds > threshold).float()
    TP = (preds * labels).sum(dim=0)
    FN = ((1 - preds) * labels).sum(dim=0)
    recall = TP / (TP + FN + 1e-8)
    return recall.mean().item()


def precision(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> float:
    """
    Computes macro-averaged precision across all classes.

    Args:
        preds (torch.Tensor): Model predictions after sigmoid, shape (B, C).
        labels (torch.Tensor): Ground truth binary labels, shape (B, C).
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Macro-averaged precision.
    """
    preds = (preds > threshold).float()
    TP = (preds * labels).sum(dim=0)
    FP = (preds * (1 - labels)).sum(dim=0)
    prec = TP / (TP + FP + 1e-8)
    return prec.mean().item()


def f1_score(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> float:
    """
    Computes macro-averaged F1-score across all classes.

    Args:
        preds (torch.Tensor): Model predictions after sigmoid, shape (B, C).
        labels (torch.Tensor): Ground truth binary labels, shape (B, C).
        threshold (float): Threshold to binarize predictions.

    Returns:
        float: Macro-averaged F1-score.
    """
    prec = precision(preds, labels, threshold)
    rec = sensitivity(preds, labels, threshold)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def macro_metrics(preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.3) -> dict:
    """
    Computes a dictionary of macro-averaged metrics.

    Args:
        preds (torch.Tensor): Model predictions after sigmoid, shape (B, C).
        labels (torch.Tensor): Ground truth binary labels, shape (B, C).
        threshold (float): Threshold to binarize predictions.

    Returns:
        dict: Dictionary with keys: 'accuracy', 'sensitivity', 'precision', 'f1_score'.
    """
    return {
        "accuracy": accuracy(preds, labels, threshold),
        "sensitivity": sensitivity(preds, labels, threshold),
        "precision": precision(preds, labels, threshold),
        "f1_score": f1_score(preds, labels, threshold),
    }


class PerClassMetrics:
    """
    Computes per-class metrics (precision, sensitivity, F1-score, accuracy)
    for multi-label classification tasks. Supports batch-wise updates and
    aggregation over multiple steps (e.g., across an epoch).
    """

    def __init__(self, num_classes: int):
        self.num_classes = num_classes
        self.reset()

    def reset(self):
        self.tp = torch.zeros(self.num_classes)
        self.fp = torch.zeros(self.num_classes)
        self.fn = torch.zeros(self.num_classes)
        self.tn = torch.zeros(self.num_classes)

    def update(self, preds: torch.Tensor, labels: torch.Tensor, threshold: float = 0.7):
        preds = (preds > threshold).float()
        for i in range(self.num_classes):
            pred_i = preds[:, i]
            label_i = labels[:, i]
            self.tp[i] += ((pred_i == 1) & (label_i == 1)).sum().item()
            self.fp[i] += ((pred_i == 1) & (label_i == 0)).sum().item()
            self.fn[i] += ((pred_i == 0) & (label_i == 1)).sum().item()
            self.tn[i] += ((pred_i == 0) & (label_i == 0)).sum().item()

    def compute_metrics(self) -> dict:
        precision = self.tp / (self.tp + self.fp + 1e-8)
        sensitivity = self.tp / (self.tp + self.fn + 1e-8)
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity + 1e-8)
        accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn + 1e-8)

        return {
            "precision": precision.tolist(),
            "sensitivity": sensitivity.tolist(),
            "f1_score": f1.tolist(),
            "accuracy": accuracy.tolist()
        }
