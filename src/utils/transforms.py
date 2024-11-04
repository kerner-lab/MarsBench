from torchvision import transforms


def get_transforms(cfg):
    task = cfg.task
    if hasattr(cfg.model.get(task, None), "input_size"):
        image_size = tuple(cfg.model.get(task).input_size)[
            1:
        ]  # Remove channel dimension
    else:
        image_size = tuple(cfg.transforms.image_size)

    # Mean and std for normalization
    mean = cfg.transforms.mean
    std = cfg.transforms.std

    # Training transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomApply([transforms.RandomRotation(90)], p=0.25),
            transforms.RandomApply([transforms.RandomRotation(180)], p=0.25),
            transforms.RandomApply([transforms.RandomRotation(270)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    return train_transform, val_transform
