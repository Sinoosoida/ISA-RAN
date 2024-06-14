import os
import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()s

    def forward(self, inputs, targets, smooth=1):
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2. * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)
        return 1 - dice


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        channels = in_channels
        for i in range(num_layers):
            self.layers.append(nn.Sequential(
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, growth_rate, kernel_size=3, padding=1, bias=False),
                nn.Dropout(0.1)
            ))
            channels += growth_rate

    def forward(self, x):
        for layer in self.layers:
            new_features = layer(x)
            x = torch.cat([x, new_features], 1)
        return x


class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding='same'),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.transition(x)


class UpConcat(nn.Module):
    def __init__(self, skip_channels, input_channels, output_channels):
        super(UpConcat, self).__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels + output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

    def forward(self, skip, x):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


def SEGMENTATION_GET_PRED(self, m):
    return m > 0.5


class UNetGPT(pl.LightningModule):
    def __init__(self, name: str = 'pred', loss_fn=DiceLoss(), lr=5e-4, dropout=0.0, weight_decay=1e-8, gamma=0.9985,
                 residual=False, down=[1, 1, 18, 1, 5], up=[2, 1, 6, 1]):
        super(UNetGPT, self).__init__()
        self.name = name.replace(" ", "_")
        self.loss_fn = loss_fn
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.save_hyperparameters()

        # Calculate scale factor based on the maximum channels in `down`
        scale_factor = max(down) / 512

        # Downward path
        self.down1 = DenseBlock(down[0], int(16 * scale_factor), 2)
        self.trans1 = TransitionLayer(down[0] + int(16 * scale_factor) * 2, down[1])

        self.down2 = DenseBlock(down[1], int(32 * scale_factor), 2)
        self.trans2 = TransitionLayer(down[1] + int(32 * scale_factor) * 2, down[2])

        self.down3 = DenseBlock(down[2], int(64 * scale_factor), 2)
        self.trans3 = TransitionLayer(down[2] + int(64 * scale_factor) * 2, down[3])

        self.down4 = DenseBlock(down[3], int(128 * scale_factor), 2)
        self.trans4 = TransitionLayer(down[3] + int(128 * scale_factor) * 2, down[4])

        # Bottleneck
        self.bottleneck = DenseBlock(down[4], int(256 * scale_factor), 2)

        # Upward path
        self.up3 = UpConcat(down[3] + int(128 * scale_factor) * 2, down[4] + int(256 * scale_factor) * 2, up[0])
        self.up2 = UpConcat(down[2] + int(64 * scale_factor) * 2, up[0], up[1])
        self.up1 = UpConcat(down[1] + int(32 * scale_factor) * 2, up[1], up[2])
        self.up0 = UpConcat(down[0] + int(16 * scale_factor) * 2, up[2], up[3])

        self.final_conv = nn.Conv2d(up[3], 1, kernel_size=1)
        self.activation = nn.Sigmoid()

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

    get_pred = SEGMENTATION_GET_PRED


# Загрузка модели
model = torch.load("./model_state.pth")


# Функция для обработки изображения
def image_processing(name: str, is_img: bool = True) -> torch.Tensor:
    if name.lower().endswith('.tiff') or name.lower().endswith('.tif'):
        return torch.Tensor(np.load(name))
    else:
        image = Image.open(name).convert('L')
        image = np.array(image)
        return torch.Tensor(image)


def load_and_preprocess_image(image_path: str, image_processing, T=None, T_x=None) -> torch.Tensor:
    # Загрузка изображения
    image = image_processing(image_path, is_img=True)

    image_shape = image.shape

    if len(image.shape) == 2:
        image = image.unsqueeze(2)

    # Применение аугментаций, специфичных для x
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

    # Преобразование x в NumPy для Albumentations
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


# Функция для сохранения изображения
def save_image(tensor, path):
    tensor = tensor.detach()  # Отсоединяем тензор от графа вычислений
    image = tensor.squeeze().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    img = Image.fromarray(image)
    img.save(path)


# Функция для наложения изображений
def overlay_images(img1, img2, alpha=0.5):
    img1 = img1.squeeze().cpu().numpy()
    img2 = img2.detach().squeeze().cpu().numpy()  # Отсоединяем тензор от графа вычислений
    img1 = (img1 * 255).astype(np.uint8)
    img2 = (img2 * 255).astype(np.uint8)

    img1 = Image.fromarray(img1).convert("RGBA")
    img2 = Image.fromarray(img2).convert("RGBA")

    blended = Image.blend(img1, img2, alpha=alpha)
    return blended


# Определение трансформаций
test_transforms = A.Compose([
    A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=2 ** 5, pad_width_divisor=2 ** 5, border_mode=0,
                  value=0, mask_value=0, always_apply=True),
    ToTensorV2(),
])

x_transforms = A.Compose([
    # A.GaussNoise(var_limit=(500), mean=0, p=1),
    ToTensorV2(),
])

# Обработка всех изображений в директории
image_dir = "./raw_images"
output_dir = "./segmented_images"
os.makedirs(output_dir, exist_ok=True)

for filename in os.listdir(image_dir):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif')):
        image_path = os.path.join(image_dir, filename)
        processed_image = load_and_preprocess_image(image_path, image_processing, T=test_transforms,
                                                    T_x=x_transforms).unsqueeze(0).unsqueeze(0)

        # Получение предсказания модели
        output = model(processed_image)

        # Создание и сохранение изображений
        base_filename = os.path.splitext(filename)[0]
        save_image(processed_image, os.path.join(output_dir, f"{base_filename}_original.png"))
        save_image(output, os.path.join(output_dir, f"{base_filename}_predicted.png"))

        overlayed_image = overlay_images(processed_image, output)
        overlayed_image.save(os.path.join(output_dir, f"{base_filename}_overlayed.png"))

print("Images processed and saved successfully.")
