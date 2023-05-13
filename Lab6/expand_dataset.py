import os
from PIL import Image

SOURCE_PATH = 'dataset/train/other'
DEST_PATH = 'train/other'

for path in os.listdir(SOURCE_PATH):
  image = Image.open(f'{SOURCE_PATH}/{path}')
  for deg in range(0, 360, 45):
    rotated = image.rotate(deg)
    rotated.save(f'{DEST_PATH}/{deg}_{path}')
