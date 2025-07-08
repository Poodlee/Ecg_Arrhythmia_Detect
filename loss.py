import torch
import torch.nn as nn
import torch.nn.functional as F


class LossFactory:
    def __init__(self, loss_type, alpha, gamma, pos_weight, class_weights):
        """
        loss_type: 'bce', 'focal', 'asymmetric'
        alpha, gamma: focal loss parameter
        pos_weight: for BCEWithLogitsLoss to handle imbalance (Tensor of shape [num_classes])
        class_weights: optional per-class weights for asymmetric loss
        """
        self.loss_type = loss_type
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.class_weights = class_weights

    def get_loss(self):
        if self.loss_type == 'bce':
            return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        elif self.loss_type == 'focal':
            return self.focal_loss
        elif self.loss_type == 'asymmetric':
            return self.asymmetric_loss
        else:
            raise ValueError(f"Unsupported loss type: {self.loss_type}")

    def focal_loss(self, inputs, targets):
        """
        Multilabel focal loss.
        inputs: logits (before sigmoid), shape (B, C)
        targets: binary labels, shape (B, C)
        """
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = targets * probs + (1 - targets) * (1 - probs)  # pt = p_t
        focal_term = (1 - pt) ** self.gamma
        loss = self.alpha * focal_term * bce_loss
        return loss.mean()

    def asymmetric_loss(self, inputs, targets, gamma_neg=4, gamma_pos=1):
        """
        Asymmetric loss (BCE-based) for extreme class imbalance.
        Computes Asymmetric Loss for multi-label classification with extreme class imbalance.

    This loss function is an enhancement of Focal Loss that treats positive and negative 
    classes differently by applying separate focusing parameters (`gamma_pos` and `gamma_neg`).
    It reduces the contribution of well-classified negative examples while emphasizing 
    hard-to-classify positive examples. This behavior is particularly useful in cases 
    where positive labels are rare and false negatives are costly.

    Reference:
        Ridnik et al., "Asymmetric Loss For Multi-Label Classification" 
        (https://arxiv.org/abs/2009.14119)

    Args:
        inputs (torch.Tensor): Logits before sigmoid, shape (B, C).
        targets (torch.Tensor): Binary ground-truth labels, shape (B, C).
        gamma_neg (float): Focusing parameter for negative class (default: 4).
        gamma_pos (float): Focusing parameter for positive class (default: 1).

    Returns:
        torch.Tensor: Scalar tensor representing the mean asymmetric loss.

    Behavior:
        - When gamma_neg is large, well-predicted negatives are down-weighted heavily.
        - Positive examples (targets == 1) are preserved for learning to reduce false negatives.
        - Especially suited for highly imbalanced multi-label settings (e.g., rare diseases).
        
        Referenced from: https://arxiv.org/abs/2009.14119
        
        """
        x_sigmoid = torch.sigmoid(inputs)
        xs_pos = x_sigmoid
        xs_neg = 1.0 - x_sigmoid

        # Calculate basic CE loss
        loss = targets * torch.log(xs_pos + 1e-8) + (1 - targets) * torch.log(xs_neg + 1e-8)

        # Asymmetric focusing
        if gamma_neg > 0 or gamma_pos > 0:
            pt = targets * xs_pos + (1 - targets) * xs_neg
            gamma = targets * gamma_pos + (1 - targets) * gamma_neg
            focal_term = (1 - pt) ** gamma
            loss *= focal_term

        # Class weights if given
        if self.class_weights is not None:
            loss *= self.class_weights

        return -loss.mean()
    
def bce(self, inputs, targets):
    """
    Binary Cross Entropy loss with logits.
    inputs: logits (before sigmoid), shape (B, C)
    targets: binary labels, shape (B, C)
    """
    return nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)(inputs, targets)