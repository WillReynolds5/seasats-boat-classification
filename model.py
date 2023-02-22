import torch.nn as nn
from vit_pytorch import SimpleViT

# TODO: research what model arch will be optimal for seasats onboard hardware

class ViT(nn.Module):
    """
    Implementation of the state of the art the Vision Transformer (ViT) model, tuned for the `SEASAT` dataset.

    The module uses the SimpleViT class from the vit_pytorch library. The ViT model
    takes an input image and transforms it into a sequence of patches, which are then processed by a
    multi-layer transformer network as a 1d sequence of pixels. The final output of the network is a
    single vector of size [2,1 ], which classifies images as - contains a boat / doesnt contain a boat.

    Args:
        image_size (int): The size of the input image.
        patch_size (int): The size of the image patches.
        num_classes (int): The number of output classes.
        dim (int): The dimension of the transformer embedding.
        depth (int): The number of transformer layers.
        heads (int): The number of attention heads in each transformer layer.
        mlp_dim (int): The dimension of the multi-layer perceptron in each transformer layer.
    """
    def __init__(self, image_size=256, patch_size=32, num_classes=2, dim=1024, depth=3, heads=8, mlp_dim=1024):
        super().__init__()

        self.vit = SimpleViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim
        )

    def forward(self, x):
        """
        forward pass of the ViT model.

        Args:
            x (torch.Tensor): The input tensor, with shape (batch_size, num_channels, height, width).

        Returns:
            The output tensor, with shape (batch_size, num_classes).
        """
        x = self.vit(x)
        return x
