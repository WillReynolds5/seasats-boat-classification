import argparse
import torch
from PIL import Image

from preprocess_data import preprocess_data
from model import ViT

def evaluate(model, image_path):
    """
    Evaluates the given model on the given image.

    Args:
        model (nn.Module): The model to evaluate.
        image_path (str): The path to the image file to evaluate.

    Returns:
        The model's prediction for the given image.
    """
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = preprocess_data(image)
    image = image.unsqueeze(0) # Add batch dimension
    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = "boat" if predicted.item() == 1 else "not boat"
        print(f"Prediction for image {image_path}: {prediction}")
        return prediction


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate a ship image classification model.")
    parser.add_argument("image_path", type=str, help="Path to image to evaluate.")
    parser.add_argument("--model_path", type=str, default="model_checkpoints/model.pth", help="Path to saved model file.")
    args = parser.parse_args()

    model = ViT()
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint)
    evaluate(model, args.image_path)
