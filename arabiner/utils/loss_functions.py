import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional
import torch.autograd as autograd

def mcc_loss(outputs_target, temperature=2, class_num=4):
    train_bs = outputs_target.size(0)
    outputs_target_temp = outputs_target / temperature
    target_softmax_out_temp = nn.Softmax(dim=1)(outputs_target_temp)
    target_entropy_weight = Entropy(target_softmax_out_temp).detach()
    target_entropy_weight = 1 + torch.exp(-target_entropy_weight)
    target_entropy_weight = train_bs * target_entropy_weight / torch.sum(target_entropy_weight)
    cov_matrix_t = target_softmax_out_temp.mul(target_entropy_weight.view(-1, 1)).transpose(1, 0).mm(target_softmax_out_temp)
    cov_matrix_t = cov_matrix_t / torch.sum(cov_matrix_t, dim=1)
    mcc_loss = (torch.sum(cov_matrix_t) - torch.trace(cov_matrix_t)) / class_num
    return mcc_loss

def EntropyLoss(input_):
    # print("input_ shape", input_.shape)
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out + 1e-5)))
    return entropy / float(input_.size(0))

def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy


class GCE(nn.Module):
    def __init__(self, num_classes=4, q=0.7):
        super(GCE, self).__init__()
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-6, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()


class AdaptiveLoss(nn.Module):
    def __init__(self, weight_ce=0.5, weight_sigmoid=0.5):
        super(AdaptiveLoss, self).__init__()
        self.weight_ce = weight_ce
        self.weight_sigmoid = weight_sigmoid
        self.w = nn.Parameter(torch.tensor(2.0))

    def forward(self, logits, targets, epoch):
        """
        Forward pass of the modified loss function with adaptive weighting.

        Parameters:
        - logits (Tensor): 2D tensor of shape (batch_size, num_classes) representing the predicted logits.
        - targets (Tensor): 1D tensor of shape (batch_size,) representing the ground truth labels.
        - epoch (int): The current epoch number.

        Returns:
        - loss (Tensor): Scalar tensor representing the modified loss.
        """
        device = logits.device  # Get the device of the logits tensor

        # Convert targets to one-hot encoding
        num_classes = logits.size(1)
        targets_onehot = F.one_hot(targets, num_classes=num_classes).float()

        # Move the parameter w to the same device as the logits tensor
        w = self.w.to(device)

        # Compute the sigmoid term
        sigmoid_logits = torch.sigmoid(logits)
        sigmoid_term = torch.sum(targets_onehot * sigmoid_logits)

        # Compute the cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets)

        # Compute the beta value
        beta = torch.exp(-epoch / w)

        # Compute the final loss
        loss = self.weight_ce * ce_loss + self.weight_sigmoid * sigmoid_term * beta

        return loss


class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, smooth=1e-6):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.smooth = smooth

    def forward(self, logits, labels):
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1))

        # Calculate true positives, false positives, and false negatives
        tp = torch.sum(probs * labels_onehot, dim=0)
        fp = torch.sum(probs * (1 - labels_onehot), dim=0)
        fn = torch.sum((1 - probs) * labels_onehot, dim=0)

        # Calculate Tversky coefficient
        tversky_coeff = (tp + self.smooth) / (tp + self.alpha * fp + (1 - self.alpha) * fn + self.smooth)

        # Average class scores
        tversky_loss = 1 - torch.mean(tversky_coeff)

        return tversky_loss

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1))

        # Calculate focal weights
        focal_weights = torch.where(labels_onehot == 1, 1 - probs, probs)
        focal_weights = (focal_weights.pow(self.gamma)).detach()

        # Calculate focal loss
        loss = F.nll_loss(log_probs, labels, weight=focal_weights, reduction=self.reduction)

        return loss
    
class DiceLoss(nn.Module):
    """
    Dice coefficient for short, is an F1-oriented statistic used to gauge the similarity of two sets.
    Given two sets A and B, the vanilla dice coefficient between them is given as follows:
        Dice(A, B)  = 2 * True_Positive / (2 * True_Positive + False_Positive + False_Negative)
                    = 2 * |A and B| / (|A| + |B|)

    Math Function:
        U-NET: https://arxiv.org/abs/1505.04597.pdf
        dice_loss(p, y) = 1 - numerator / denominator
            numerator = 2 * \sum_{1}^{t} p_i * y_i + smooth
            denominator = \sum_{1}^{t} p_i + \sum_{1} ^{t} y_i + smooth
        if square_denominator is True, the denominator is \sum_{1}^{t} (p_i ** 2) + \sum_{1} ^{t} (y_i ** 2) + smooth
        V-NET: https://arxiv.org/abs/1606.04797.pdf
    Args:
        smooth (float, optional): a manual smooth value for numerator and denominator.
        square_denominator (bool, optional): [True, False], specifies whether to square the denominator in the loss function.
        with_logits (bool, optional): [True, False], specifies whether the input tensor is normalized by Sigmoid/Softmax funcs.
        ohem_ratio: max ratio of positive/negative, defautls to 0.0, which means no ohem.
        alpha: dsc alpha
    Shape:
        - input: (*)
        - target: (*)
        - mask: (*) 0,1 mask for the input sequence.
        - Output: Scalar loss
    Examples:
        >>> loss = DiceLoss(with_logits=True, ohem_ratio=0.1)
        >>> input = torch.FloatTensor([2, 1, 2, 2, 1])
        >>> input.requires_grad=True
        >>> target = torch.LongTensor([0, 1, 0, 0, 0])
        >>> output = loss(input, target)
        >>> output.backward()
    """
    def __init__(self,
                 smooth: Optional[float] = 1e-4,
                 square_denominator: Optional[bool] = False,
                 with_logits: Optional[bool] = True,
                 ohem_ratio: float = 0.0,
                 alpha: float = 0.0,
                 reduction: Optional[str] = "mean",
                 index_label_position=True) -> None:
        super(DiceLoss, self).__init__()

        self.reduction = reduction
        self.with_logits = with_logits
        self.smooth = smooth
        self.square_denominator = square_denominator
        self.ohem_ratio = ohem_ratio
        self.alpha = alpha
        self.index_label_position = index_label_position

    def forward(self, input: Tensor, target: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits_size = input.shape[-1]

        if logits_size != 1:
            loss = self._multiple_class(input, target, logits_size, mask=mask)
        else:
            loss = self._binary_class(input, target, mask=mask)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

    def _compute_dice_loss(self, flat_input, flat_target):
        flat_input = ((1 - flat_input) ** self.alpha) * flat_input
        interection = torch.sum(flat_input * flat_target, -1)
        if not self.square_denominator:
            loss = 1 - ((2 * interection + self.smooth) /
                        (flat_input.sum() + flat_target.sum() + self.smooth))
        else:
            loss = 1 - ((2 * interection + self.smooth) /
                        (torch.sum(torch.square(flat_input, ), -1) + torch.sum(torch.square(flat_target), -1) + self.smooth))

        return loss

    def _multiple_class(self, input, target, logits_size, mask=None):
        flat_input = input
        flat_target = F.one_hot(target, num_classes=logits_size).float() if self.index_label_position else target.float()
        flat_input = torch.nn.Softmax(dim=1)(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        loss = None
        if self.ohem_ratio > 0 :
            mask_neg = torch.logical_not(mask)
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                neg_example = target != label_idx

                pos_num = pos_example.sum()
                neg_num = mask.sum() - (pos_num - (mask_neg & pos_example).sum())
                keep_num = min(int(pos_num * self.ohem_ratio / logits_size), neg_num)

                if keep_num > 0:
                    neg_scores = torch.masked_select(flat_input, neg_example.view(-1, 1).bool()).view(-1, logits_size)
                    neg_scores_idx = neg_scores[:, label_idx]
                    neg_scores_sort, _ = torch.sort(neg_scores_idx, )
                    threshold = neg_scores_sort[-keep_num + 1]
                    cond = (torch.argmax(flat_input, dim=1) == label_idx & flat_input[:, label_idx] >= threshold) | pos_example.view(-1)
                    ohem_mask_idx = torch.where(cond, 1, 0)

                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                    flat_input_idx = flat_input_idx * ohem_mask_idx
                    flat_target_idx = flat_target_idx * ohem_mask_idx
                else:
                    flat_input_idx = flat_input[:, label_idx]
                    flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

        else:
            for label_idx in range(logits_size):
                pos_example = target == label_idx
                flat_input_idx = flat_input[:, label_idx]
                flat_target_idx = flat_target[:, label_idx]

                loss_idx = self._compute_dice_loss(flat_input_idx.view(-1, 1), flat_target_idx.view(-1, 1))
                if loss is None:
                    loss = loss_idx
                else:
                    loss += loss_idx
            return loss

    def _binary_class(self, input, target, mask=None):
        flat_input = input.view(-1)
        flat_target = target.view(-1).float()
        flat_input = torch.sigmoid(flat_input) if self.with_logits else flat_input

        if mask is not None:
            mask = mask.float()
            flat_input = flat_input * mask
            flat_target = flat_target * mask
        else:
            mask = torch.ones_like(target)

        if self.ohem_ratio > 0:
            pos_example = target > 0.5
            neg_example = target <= 0.5
            mask_neg_num = mask <= 0.5

            pos_num = pos_example.sum() - (pos_example & mask_neg_num).sum()
            neg_num = neg_example.sum()
            keep_num = min(int(pos_num * self.ohem_ratio), neg_num)

            neg_scores = torch.masked_select(flat_input, neg_example.bool())
            neg_scores_sort, _ = torch.sort(neg_scores, )
            threshold = neg_scores_sort[-keep_num+1]
            cond = (flat_input > threshold) | pos_example.view(-1)
            ohem_mask = torch.where(cond, 1, 0)
            flat_input = flat_input * ohem_mask
            flat_target = flat_target * ohem_mask

        return self._compute_dice_loss(flat_input, flat_target)

    def __str__(self):
        return f"Dice Loss smooth:{self.smooth}, ohem: {self.ohem_ratio}, alpha: {self.alpha}"

    def __repr__(self):
        return str(self)

class UnifiedLoss2(nn.Module):
    def __init__(self, weight_dice=0.2, weight_tversky=0.6, weight_focal=0.2, weight_ce=1.0, alpha=0.5, gamma=2, reduction='mean'):
        super(UnifiedLoss2, self).__init__()
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.weight_ce = weight_ce
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.dice_loss = DiceLoss(smooth=1, ohem_ratio=0.3, alpha=0.01)

    def forward(self, logits, labels):
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)
        
        #print(logits.shape)
        #print(labels.detach().cpu().numpy().tolist())
        #print(labels.shape)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).to(logits.device)
        #print('OHE', labels_onehot)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(probs, labels_onehot)


        # Calculate Tversky Loss
        tversky_loss = self.tversky_loss(probs, labels_onehot)


        # Calculate Focal Loss
        focal_loss = self.focal_loss(logits, labels)


        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, labels, reduction=self.reduction)

        # Combine the losses
        unified_loss = (
            self.weight_dice * dice_loss +
            self.weight_tversky * tversky_loss +
            self.weight_focal * focal_loss +
            self.weight_ce * ce_loss
        )

        return unified_loss


    def tversky_loss(self, probs, labels_onehot):
        tp = torch.sum(probs * labels_onehot, dim=0)
        fp = torch.sum(probs * (1 - labels_onehot), dim=0)
        fn = torch.sum((1 - probs) * labels_onehot, dim=0)
        tversky_coeff = (tp + 1e-6) / (tp + self.alpha * fp + (1 - self.alpha) * fn + 1e-6)
        if self.reduction == 'mean':
            tversky_loss = 1 - torch.mean(tversky_coeff)
        else:
            tversky_loss = 1 - tversky_coeff
        return tversky_loss

    def focal_loss(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weights = (1 - pt).pow(self.gamma)
        if self.reduction == 'mean':
            focal_loss = (focal_weights * ce_loss).mean()
        else:
            focal_loss = focal_weights * ce_loss
        return focal_loss

class sigmoidF1(nn.Module):

    def __init__(self, S = -10, E = 1):
        super(sigmoidF1, self).__init__()
        self.S = S
        self.E = E

    @torch.cuda.amp.autocast()
    def forward(self, y_hat, y):
        
        y_hat = torch.sigmoid(y_hat)

        b = torch.tensor(self.S).to(y_hat.device)
        c = torch.tensor(self.E).to(y_hat.device)

        sig = 1 / (1 + torch.exp(b * (y_hat + c)))

        tp = torch.sum(sig * y, dim=0)
        fp = torch.sum(sig * (1 - y), dim=0)
        fn = torch.sum((1 - sig) * y, dim=0)

        sigmoid_f1 = 2*tp / (2*tp + fn + fp + 1e-16)
        cost = 1 - sigmoid_f1
        macroCost = torch.mean(cost)

        return macroCost

class UnifiedLoss(nn.Module):
    def __init__(self, weight_dice=0.2, weight_tversky=0.2, weight_focal=0.2, weight_ce=0.4, alpha=0.5, gamma=2, reduction='mean'):
        super(UnifiedLoss, self).__init__()
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.weight_ce = weight_ce
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    @staticmethod
    def irm_penalty(logits, y):
        device = logits[0][0].device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result
    
    def forward(self, logits, labels):
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)
    
        #print(logits.shape)
        #print(labels.detach().cpu().numpy().tolist())
        #print(labels.shape)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).to(logits.device)
        #print('OHE', labels_onehot)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(probs, labels_onehot)


        # Calculate Tversky Loss
        tversky_loss = self.tversky_loss(probs, labels_onehot)

        # Calculate Focal Loss
        focal_loss = self.focal_loss(logits, labels)
    

        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, labels, reduction=self.reduction) + 5 * self.irm_penalty(logits, labels)

        # Combine the losses
        unified_loss = (
            self.weight_dice * dice_loss +
            self.weight_tversky * tversky_loss +
            self.weight_focal * focal_loss +
            self.weight_ce * ce_loss 
        )

        return unified_loss

    def dice_loss(self, probs, labels_onehot):
        intersection = torch.sum(probs * labels_onehot, dim=0)
        union = torch.sum(probs, dim=0) + torch.sum(labels_onehot, dim=0)
        dice_coeff = (2 * intersection + 1e-6) / (union + 1e-6)
        if self.reduction == 'mean':
            dice_loss = 1 - torch.mean(dice_coeff)
        else:
            dice_loss = 1 - dice_coeff
        return dice_loss

    def tversky_loss(self, probs, labels_onehot):
        tp = torch.sum(probs * labels_onehot, dim=0)
        fp = torch.sum(probs * (1 - labels_onehot), dim=0)
        fn = torch.sum((1 - probs) * labels_onehot, dim=0)
        tversky_coeff = (tp + 1e-6) / (tp + self.alpha * fp + (1 - self.alpha) * fn + 1e-6)
        if self.reduction == 'mean':
            tversky_loss = 1 - torch.mean(tversky_coeff)
        else:
            tversky_loss = 1 - tversky_coeff
        return tversky_loss

    def focal_loss(self, logits, labels):
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weights = (1 - pt).pow(self.gamma)
        if self.reduction == 'mean':
            focal_loss = (focal_weights * ce_loss).mean()
        else:
            focal_loss = focal_weights * ce_loss
        return focal_loss
    

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight_focal=1.0, weight_ce=1.0, alpha=0.5, beta=0.5, gamma=2, smooth=1e-6):
        super(FocalTverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.weight_focal = weight_focal
        self.weight_ce = weight_ce
        
    @staticmethod
    def irm_penalty(logits, y):
        device = logits[0][0].device
        scale = torch.tensor(1.).to(device).requires_grad_()
        loss_1 = F.cross_entropy(logits[::2] * scale, y[::2])
        loss_2 = F.cross_entropy(logits[1::2] * scale, y[1::2])
        grad_1 = autograd.grad(loss_1, [scale], create_graph=True)[0]
        grad_2 = autograd.grad(loss_2, [scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def forward(self, logits, targets):
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)
        # Convert targets to one-hot encoding
        targets_onehot = F.one_hot(targets, num_classes=logits.size(1))

        # Compute the Tversky coefficient
        tp = torch.sum(probs * targets_onehot, dim=0)
        fp = torch.sum(probs * (1 - targets_onehot), dim=0)
        fn = torch.sum((1 - probs) * targets_onehot, dim=0)
        tversky_coeff = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        # Compute the Focal Tversky loss
        focal_tversky_loss = torch.pow((1 - tversky_coeff), self.gamma) + 5 * self.irm_penalty(logits, targets)

        return self.weight_focal * focal_tversky_loss.mean() + self.weight_ce * F.cross_entropy(logits, targets) 



class UnifiedLossMT(nn.Module):
    def __init__(self, weight_dice=0.2, weight_tversky=0.2, weight_focal=0.2, weight_ce=1.0, alpha=0.5, gamma=2, reduction='mean'):
        super(UnifiedLossMT, self).__init__()
        self.weight_dice = weight_dice
        self.weight_tversky = weight_tversky
        self.weight_focal = weight_focal
        self.weight_ce = weight_ce
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, labels):
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)
        
        #print(logits.shape)
        #print(labels.detach().cpu().numpy().tolist())
        #print(labels.shape)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).to(logits.device)
        #print('OHE', labels_onehot)

        # Calculate Dice Loss
        dice_loss = self.dice_loss(probs, labels_onehot)
        #print('dice loss', dice_loss.shape)

        # Calculate Tversky Loss
        tversky_loss = self.tversky_loss(probs, labels_onehot)
        #print('tversky_loss', tversky_loss.shape)

        # Calculate Focal Loss
        focal_loss = self.focal_loss(logits, labels)
        #print('focal_loss', focal_loss.shape)

        # Calculate Cross Entropy Loss
        ce_loss = F.cross_entropy(logits, labels, reduction='mean')
        #print('ce_loss', ce_loss.shape)

        # Combine the losses
        unified_loss = (
            self.weight_dice * dice_loss +
            self.weight_tversky * tversky_loss +
            self.weight_focal * focal_loss +
            self.weight_ce * ce_loss
        )

        return unified_loss

    def dice_loss(self, probs, labels_onehot):
        intersection = torch.sum(probs * labels_onehot, dim=0)
        union = torch.sum(probs, dim=0) + torch.sum(labels_onehot, dim=0)
        dice_coeff = (2 * intersection + 1e-6) / (union + 1e-6)
        if self.reduction == 'mean':
            dice_loss = 1 - torch.mean(dice_coeff)
        else:
            dice_loss = 1 - dice_coeff
        return dice_loss

    def tversky_loss(self, probs, labels_onehot):
        tp = torch.sum(probs * labels_onehot, dim=0)
        fp = torch.sum(probs * (1 - labels_onehot), dim=0)
        fn = torch.sum((1 - probs) * labels_onehot, dim=0)
        tversky_coeff = (tp + 1e-6) / (tp + self.alpha * fp + (1 - self.alpha) * fn + 1e-6)
        if self.reduction == 'mean':
            tversky_loss = 1 - torch.mean(tversky_coeff)
        else:
            tversky_loss = 1 - tversky_coeff
        return tversky_loss

    def focal_loss(self, logits, labels, reduction='mean'):
        log_probs = F.log_softmax(logits, dim=1)
        ce_loss = F.nll_loss(log_probs, labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_weights = (1 - pt).pow(self.gamma)
        if reduction == 'mean':
            focal_loss = (focal_weights * ce_loss).mean()
        else:
            focal_loss = focal_weights * ce_loss
        return focal_loss



class SymmetricFocalTverskyLoss(nn.Module):
    def __init__(self, delta=0.7, gamma=0.75):
        super(SymmetricFocalTverskyLoss, self).__init__()
        self.delta = delta
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        Forward pass of the Symmetric Focal Tversky loss.

        Parameters:
        - logits (Tensor): 2D tensor of shape (batch_size, num_classes) representing the predicted logits.
        - labels (Tensor): 1D tensor of shape (batch_size,) representing the ground truth labels.

        Returns:
        - loss (Tensor): Scalar tensor representing the symmetric focal Tversky loss.
        """
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).float()

        # Calculate true positives (tp), false negatives (fn), and false positives (fp)
        tp = torch.sum(probs * labels_onehot, dim=0)
        fn = torch.sum(labels_onehot * (1 - probs), dim=0)
        fp = torch.sum((1 - labels_onehot) * probs, dim=0)

        # Calculate Tversky coefficients
        tversky_coeff = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - tversky_coeff[0]) * torch.pow(1 - tversky_coeff[0], -self.gamma)
        fore_dice = (1 - tversky_coeff[1]) * torch.pow(1 - tversky_coeff[1], -self.gamma)

        # Average class scores
        loss = torch.mean(torch.stack([back_dice, fore_dice]))

        return loss


class SymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=0.5):
        super(SymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        Forward pass of the Symmetric Unified Focal loss.

        Parameters:
        - logits (Tensor): 2D tensor of shape (batch_size, num_classes) representing the predicted logits.
        - labels (Tensor): 1D tensor of shape (batch_size,) representing the ground truth labels.

        Returns:
        - loss (Tensor): Scalar tensor representing the symmetric unified focal loss.
        """
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).float()

        # Calculate true positives (tp), false negatives (fn), and false positives (fp)
        tp = torch.sum(probs * labels_onehot, dim=0)
        fn = torch.sum(labels_onehot * (1 - probs), dim=0)
        fp = torch.sum((1 - labels_onehot) * probs, dim=0)

        # Calculate Tversky coefficients
        tversky_coeff = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)

        # Calculate losses separately for each class, enhancing both classes
        back_dice = (1 - tversky_coeff[0]) * torch.pow(1 - tversky_coeff[0], -self.gamma)
        fore_dice = (1 - tversky_coeff[1]) * torch.pow(1 - tversky_coeff[1], -self.gamma)

        # Calculate focal losses
        focal_back = self.weight * torch.pow(1 - tversky_coeff[0], self.gamma) * F.cross_entropy(logits, labels)
        focal_fore = (1 - self.weight) * torch.pow(1 - tversky_coeff[1], self.gamma) * F.cross_entropy(logits, labels)

        # Calculate the symmetric unified focal loss
        loss = back_dice + fore_dice + focal_back + focal_fore

        return loss + F.cross_entropy(input=logits, target=labels)


class AsymmetricUnifiedFocalLoss(nn.Module):
    def __init__(self, weight=0.5, delta=0.6, gamma=2):
        super(AsymmetricUnifiedFocalLoss, self).__init__()
        self.weight = weight
        self.delta = delta
        self.gamma = gamma

    def forward(self, logits, labels):
        """
        Forward pass of the Asymmetric Unified Focal loss.

        Parameters:
        - logits (Tensor): 2D tensor of shape (batch_size, num_classes) representing the predicted logits.
        - labels (Tensor): 1D tensor of shape (batch_size,) representing the ground truth labels.

        Returns:
        - loss (Tensor): Scalar tensor representing the asymmetric unified focal loss.
        """
        # Apply softmax activation to logits
        probs = F.softmax(logits, dim=1)

        # Convert labels to one-hot encoding
        labels_onehot = F.one_hot(labels, num_classes=logits.size(1)).float()

        # Calculate true positives (tp), false negatives (fn), and false positives (fp)
        tp = torch.sum(probs * labels_onehot, dim=0)
        fn = torch.sum(labels_onehot * (1 - probs), dim=0)
        fp = torch.sum((1 - labels_onehot) * probs, dim=0)

        # Calculate Tversky coefficients
        tversky_coeff = (tp + 1e-6) / (tp + self.delta * fn + (1 - self.delta) * fp + 1e-6)

        # Calculate losses separately for each class, enhancing the foreground class
        back_dice = (1 - tversky_coeff[0])
        fore_dice = (1 - tversky_coeff[1]) * torch.pow(1 - tversky_coeff[1], -self.gamma)

        # Calculate focal losses
        focal_back = self.weight * torch.pow(1 - tversky_coeff[0], self.gamma) * F.cross_entropy(logits, labels)
        focal_fore = (1 - self.weight) * torch.pow(1 - tversky_coeff[1], self.gamma) * F.cross_entropy(logits, labels)

        # Calculate the asymmetric unified focal loss
        loss = back_dice + fore_dice + focal_back + focal_fore

        return loss
