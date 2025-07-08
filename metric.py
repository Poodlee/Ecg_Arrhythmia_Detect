import torch
import torch.nn.functional as F

def accuracy_multiclass(preds_logits: torch.Tensor, labels_idx: torch.Tensor, num_classes: int) -> float:
    """
    Computes overall accuracy for multi-class classification.
    Args:
        preds_logits (torch.Tensor): Model raw logits, shape (B, C).
        labels_idx (torch.Tensor): Ground truth class indices, shape (B,).
    Returns:
        float: Overall accuracy.
    """
    
    predicted_labels = torch.argmax(preds_logits, dim=1) # Predicted Class index
    correct = (predicted_labels == labels_idx).float()
    return correct.mean().item()

def calculate_confusion_components(
    preds_logits: torch.Tensor,
    labels_idx: torch.Tensor,
    num_classes: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes True Positives (TP), False Positives (FP), and False Negatives (FN)
    for each class in a multi-class classification problem.

    Args:
        preds_logits (torch.Tensor): Model raw logits, shape (B, C).
        labels_idx (torch.Tensor): Ground truth class indices, shape (B,).
        num_classes (int): Total number of classes.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - TP (torch.Tensor): True Positives for each class, shape (C,).
            - FP (torch.Tensor): False Positives for each class, shape (C,).
            - FN (torch.Tensor): False Negatives for each class, shape (C,).
    """
    # 1. Logits를 예측된 클래스 인덱스로 변환
    predicted_labels = torch.argmax(preds_logits, dim=1)

    # 2. 실제 라벨과 예측 라벨을 원-핫 인코딩으로 변환
    # F.one_hot은 long 타입 입력을 기대합니다. labels_idx는 이미 long일 가능성이 높음.
    labels_one_hot = F.one_hot(labels_idx, num_classes=num_classes).float()
    preds_one_hot = F.one_hot(predicted_labels, num_classes=num_classes).float()

    # 3. 각 클래스별 TP, FP, FN 계산
    # TP: 실제도 1이고 예측도 1인 경우 (preds_one_hot * labels_one_hot)
    TP = (preds_one_hot * labels_one_hot).sum(dim=0) # (C,)

    # FP: 예측은 1인데 실제는 0인 경우 (preds_one_hot * (1 - labels_one_hot))
    FP = (preds_one_hot * (1 - labels_one_hot)).sum(dim=0) # (C,)

    # FN: 예측은 0인데 실제는 1인 경우 ((1 - preds_one_hot) * labels_one_hot)
    FN = ((1 - preds_one_hot) * labels_one_hot).sum(dim=0) # (C,)

    return TP, FP, FN

# --- Sensitivity (Recall) - Macro Average ---
def sensitivity_macro(preds_logits: torch.Tensor, labels_idx: torch.Tensor, num_classes: int) -> float:
    """
    Computes macro-averaged sensitivity (recall) for multi-class classification.

    Args:
        preds_logits (torch.Tensor): Model raw logits, shape (B, C).
        labels_idx (torch.Tensor): Ground truth class indices, shape (B,).
        num_classes (int): Total number of classes.

    Returns:
        float: Macro-averaged sensitivity.
    """
    TP, FP, FN = calculate_confusion_components(preds_logits, labels_idx, num_classes)
    
    # Calculate recall per class
    # Add a small epsilon to avoid division by zero
    recall_per_class = TP / (TP + FN + 1e-8)
    
    # Macro-averaged recall (mean of per-class recall)
    return recall_per_class.mean().item()

# --- Precision - Macro Average ---
def precision_macro(preds_logits: torch.Tensor, labels_idx: torch.Tensor, num_classes: int) -> float:
    """
    Computes macro-averaged precision for multi-class classification.

    Args:
        preds_logits (torch.Tensor): Model raw logits, shape (B, C).
        labels_idx (torch.Tensor): Ground truth class indices, shape (B,).
        num_classes (int): Total number of classes.

    Returns:
        float: Macro-averaged precision.
    """
    TP, FP, FN = calculate_confusion_components(preds_logits, labels_idx, num_classes)
    
    # Calculate precision per class
    prec_per_class = TP / (TP + FP + 1e-8)
    
    # Macro-averaged precision (mean of per-class precision)
    return prec_per_class.mean().item()

# --- F1-Score - Macro Average ---
def f1_score_macro(preds_logits: torch.Tensor, labels_idx: torch.Tensor, num_classes: int) -> float:
    """
    Computes macro-averaged F1-score for multi-class classification.

    Args:
        preds_logits (torch.Tensor): Model raw logits, shape (B, C).
        labels_idx (torch.Tensor): Ground truth class indices, shape (B,).
        num_classes (int): Total number of classes.

    Returns:
        float: Macro-averaged F1-score.
    """
    # Calculate macro-averaged precision and recall using the functions above
    prec = precision_macro(preds_logits, labels_idx, num_classes)
    rec = sensitivity_macro(preds_logits, labels_idx, num_classes)
    
    # Calculate F1-score
    if (prec + rec) == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


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
