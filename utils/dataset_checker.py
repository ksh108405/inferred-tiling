import PIL
import imghdr
from pathlib import Path

check_dir = '/media/seokhoon/dataset_drive/grad_proj/rgb-images/Vandalism/v_Vandalism_g01_c47/'

path = Path(check_dir).rglob("*.jpg")

for img_p in path:
    try:
        img = PIL.Image.open(img_p)
    except PIL.UnidentifiedImageError:
        print(img_p)

for img_p in path:
    if imghdr.what(img_p) is None:
        print(img_p)
