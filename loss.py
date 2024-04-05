import torch
import torch.nn as nn
import torch.nn.functional as F

class BCEDiceLoss(nn.Module):
    def __init__(self, weight_dice=2, smooth=0.1, class_weights=None):
        super(BCEDiceLoss, self).__init__()
        self.weight_dice = weight_dice
        self.smooth = smooth
        self.bce_loss = nn.BCEWithLogitsLoss()
        # self.class_weights = class_weights
        # Apply class weights to BCE Loss
        # self.bce_loss = nn.BCEWithLogitsLoss(weight=class_weights)

    def forward(self, input, target):
        # Apply sigmoid activation to convert logits to probabilities
        input_probs = torch.sigmoid(input)
        
        # Binary Cross-Entropy Loss
        bce_loss = self.bce_loss(input, target)

        # Dice Loss
        num = target.size(0)
        input_flat = input_probs.view(num, -1)
        target_flat = target.view(num, -1)
        intersection = (input_flat * target_flat).sum(dim=1)
        dice_loss = 1 - (2. * intersection + self.smooth) / (input_flat.sum(dim=1) + target_flat.sum(dim=1) + self.smooth)

        # Calculate mean of each loss across the batch dimension
        bce_loss = bce_loss.mean() # new line added to compute the mean of the bce losses
        dice_loss = dice_loss.mean() # new line added to compute the mean of the bce losses

        # Combine both losses
        combined_loss = bce_loss + self.weight_dice * dice_loss
        return combined_loss

