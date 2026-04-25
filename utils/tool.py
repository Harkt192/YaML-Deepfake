from PIL import Image
from torchvision import transforms
import torch

base_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    transforms.ToPILImage()
])

base_image = Image.open("real.jpg").convert("RGB")
new_image : Image.Image = base_transform(base_image)
new_image.save("real_aug_val.jpg")