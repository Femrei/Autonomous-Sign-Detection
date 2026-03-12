import os
import random
from PIL import Image
import numpy as np
import albumentations as A

# Augmentasyon islemleri icin donusumler tanimlanir
transform = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.RandomScale(scale_limit=0.2, p=0.5),
    A.CLAHE(p=0.2),
    A.HueSaturationValue(
        hue_shift_limit=20,
        sat_shift_limit=30,
        val_shift_limit=20,
        p=0.2
    ),
    A.GaussianBlur(blur_limit=(3, 7), p=0.3),
    A.RandomGamma(p=0.4),
])

# Girdi ve çıktı klasörleri
input_folder = "veri/train"
output_folder = "veri/train_aug"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Her goruntu icin veri artirma islemi
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    image = np.array(Image.open(img_path))

    augmented = transform(image=image)
    augmented_image = augmented["image"]

    save_path = os.path.join(output_folder, filename)
    Image.fromarray(augmented_image).save(save_path)
