import torchvision.transforms as transforms

def get_train_transforms():
    color_transforms = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2)
    ])

    geometric_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[45, 90]),
        transforms.RandomResizedCrop(size=(512, 512), scale=(0.8, 1.0))
    ])

    return transforms.Compose([
        transforms.RandomApply([color_transforms], p=1),
        transforms.RandomApply([geometric_transforms], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_eval_transforms():
    return transforms.Compose([
        transforms.RandomResizedCrop(size=(512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

