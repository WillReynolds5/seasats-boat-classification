import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from model import ViT
from preprocess_data import load_data


def train_model(train_loader, val_loader, model, epochs=10, lr=1e-4):
    """
    Trains the given model on the given pytorch DataLoader, and plots train and val loss.

    Args:
        train_loader (DataLoader): The training data.
        val_loader (DataLoader): The validation data.
        model (nn.Module): The model to train.
        epochs (int): The number of epochs to train for.
        lr (float): The learning rate for the optimizer.

    Returns:
        The trained model and the list of training losses.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        running_loss = 0.0

        for i, (inputs, labels) in enumerate(train_loader, 0):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # take average train loss for epoch
        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        with torch.no_grad():
            val_loss = 0.0
            for j, (inputs, labels) in enumerate(val_loader, 0):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

            # take average val loss for epoch
            val_loss /= len(val_loader)
            val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

    plt.plot(range(epochs), train_losses, label='Training Loss')
    plt.plot(range(epochs), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    plt.legend()
    plt.show()

    return model, train_losses

# TODO: add hyperparameter tuning and save best model

if __name__ == "__main__":

    training_data_path = "dataset/train"
    validation_data_path = "dataset/val"

    # Set up data loaders
    train_loader = load_data(training_data_path, batch_size=32)
    val_loader = load_data(validation_data_path, batch_size=32)

    # Set up model and optimizer
    model = ViT()

    # Train model
    trained_model, train_losses = train_model(train_loader, val_loader, model, epochs=10, lr=1e-4)

    # Save model
    torch.save(trained_model.state_dict(), "model_checkpoints/model.pth")
    print("Model saved!")
