import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import requests
from io import BytesIO

def build_transform():
    transform = A.Compose(
        [
            A.Resize(384, 384, interpolation=cv2.INTER_CUBIC),
            A.Normalize(
                mean=(0.48145466, 0.4578275, 0.40821073),
                std=(0.26862954, 0.26130258, 0.27577711),
            ),
            ToTensorV2(),
        ]
    )
    return transform

def load_image_from_url(url, div=5):
    img = BytesIO(requests.get(url).content)
    img = Image.open(img)
    w, h = img.size
    if min(w, h) // div > 385:
        img = img.resize((w // div, h // div))
    return img