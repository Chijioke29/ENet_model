import torch

def calculate_accuracy(predicted, target):
    _, predicted_labels = torch.max(predicted, 1)
    correct = (predicted_labels == target).float().sum()
    total = target.size(0) * target.size(1) * target.size(2)
    accuracy = correct / total
    return accuracy.item()

def calculate_confusion_matrix(predicted, target, num_classes):
    predicted = torch.argmax(predicted, dim=1).view(-1)
    target = target.view(-1)

    # Calculate the indices for confusion matrix
    indices = num_classes * target + predicted
    # indices = indices.to(torch.long) # new code line added when using BCEDice loss. Comment when using CE loss

    # Count occurrences of each index
    counts = torch.bincount(indices, minlength=num_classes**2)

    # Reshape counts to get confusion matrix
    confusion_matrix = counts.view(num_classes, num_classes)

    return confusion_matrix

def calculate_iou(confusion_matrix):
    intersection = confusion_matrix.diag()
    union = (confusion_matrix.sum(0) + confusion_matrix.sum(1)) - intersection
    iou = intersection / union
    return iou

def calculate_miou(confusion_matrix):
    iou = calculate_iou(confusion_matrix)
    miou = torch.mean(iou)
    return miou.item()

def calculate_f1_score(confusion_matrix):
    intersection = confusion_matrix.diag()
    ground_truth = confusion_matrix.sum(dim=1)
    predicted = confusion_matrix.sum(dim=0)
    
    precision = intersection / (predicted + 1e-20)  # Adding a small epsilon to avoid division by zero
    recall = intersection / (ground_truth + 1e-20)
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-20)
    
    return f1.mean().item(), precision, recall

