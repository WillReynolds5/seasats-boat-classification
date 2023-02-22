import torch
import torchvision
from torchvision import transforms, datasets


def preprocess_data(data):
    """
    Preprocesses the given data by randomly cropping and resizing.

    Args:
        data (PIL.Image): The data to preprocess.

    Returns:
        The preprocessed data.
    """
    transform = transforms.Compose([
        transforms.RandomCrop(256), # TODO: POSSIBLY CHANGE/REMOVE THIS ONCE SEEING DATASET, RANDOM CROP OR CENTER CROP COULD REMOVE SMALL SHIPS IN BACKGROUND FROM THE IMAGE
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # TODO: SET THESE VALUES TO THE MEAN AND STD OF THE SHIPS DATASET, CURRENTLY SET TO IMAGENET VALUES
    ])
    return transform(data)


def load_data(image_dir, batch_size):
    """
    Loads data from the given image directory and returns a PyTorch DataLoader object.

    Args:
        image_dir (str): The path to the image directory.
        batch_size (int): The batch size for the DataLoader.

    Returns:
        A PyTorch DataLoader object.
    """
    data = datasets.ImageFolder(image_dir, transform=preprocess_data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
    return dataloader
