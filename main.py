import os
import numpy as np
import torch
import torch.nn as nn
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image

class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        pass

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        pass

    def forward(self, x):
        return self.transition(x)


class UpConcat(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels):
        pass

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.trans1(d1))
        d3 = self.down3(self.trans2(d2))
        d4 = self.down4(self.trans3(d3))
        b = self.bottleneck(self.trans4(d4))

        u3 = self.up3(d4, b)
        u2 = self.up2(d3, u3)
        u1 = self.up1(d2, u2)
        u0 = self.up0(d1, u1)

        x = self.final_conv(u0)
        return self.activation(x)


# Model loading
model = torch.load("./model.pth", map_location=torch.device('cpu'))


# Function for image processing
def image_processing(name: str, is_img: bool = True) -> torch.Tensor:
    if name.lower().endswith('.tiff') or name.lower().endswith('.tif'):
        return torch.Tensor(np.load(name))
    else:
        image = Image.open(name).convert('L')
        image = np.array(image)
        return torch.Tensor(image)


def load_and_preprocess_image(image_path: str, image_processing, T=None, T_x=None) -> torch.Tensor:
    # Image loading
    image = image_processing(image_path, is_img=True)

    image_shape = image.shape

    if len(image.shape) == 2:
        image = image.unsqueeze(2)

    # Applying augmentations specific for x
    if T_x is not None:
        image_dtype = image.dtype

        image = image.numpy()

        augmented = T_x(image=image)
        image = augmented['image']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        if image.shape[0] == 1:
            image = image.permute(1, 2, 0)

        image = image.to(dtype=image_dtype)

    # Converting x to NumPy for Albumentations
    if T is not None:
        image_dtype = image.dtype

        image = image.numpy()

        augmented = T(image=image)
        image = augmented['image']

        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)

        image = image.to(dtype=image_dtype)

    if len(image_shape) == 2:
        image = image.squeeze()

    return image


# Function to save an image
def save_image(tensor, path):
    tensor = tensor.detach()  # Detach tensor from the computation graph
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(path)


# Function to overlay images
def overlay_images(img1, img2, alpha=0.5):
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.detach().squeeze().cpu().numpy()  # Detach tensor from the computation graph
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    img1 = Image.fromarray(img1).convert("RGBA")
    img2 = Image.fromarray(img2).convert("RGBA")

    blended = Image.blend(img1, img2, alpha=alpha)
    return blended


# Definition of transformations
test_transforms = A.Compose([
    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=2 ** 5, pad_width_divisor=2 ** 5, border_mode=0,
                  value=0, mask_value=0, always_apply=True),
    ToTensorV2(),
])

x_transforms = A.Compose([
    ToTensorV2(),
])

# Process all images in directory
image_dir = "./raw_images"
output_dir = "./segmented_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        image_path = os.path.join(image_dir, filename)
        processed_image = load_and_preprocess_image(image_path, image_processing, T=test_transforms,
                                                    T_x=x_transforms).unsqueeze(0).unsqueeze(0)

        # Getting model prediction
        output = model(processed_image)

        # Creating and saving images
        base_filename = os.path.splitext(filename)[0]
        save_image(processed_image, os.path.join(output_dir, f"{base_filename}_original.png"))
        save_image(output, os.path.join(output_dir, f"{base_filename}_predicted.png"))

        overlayed_image = overlay_images(processed_image, output)
        overlayed_image.save(os.path.join(output_dir, f"{base_filename}_overlayed.png"))

print("Images processed and saved successfully.")
