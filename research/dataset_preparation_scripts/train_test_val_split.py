from glob import glob
import shutil
from pathlib import Path
from tqdm import tqdm
from random import shuffle
from os.path import basename


"""
Превращает датасет, данный организаторами, в camvid формат для подгрузки на CVAT
"""


IMGS_FOLDER = "train_dataset/cv_open_dataset/open_img"
TEST_COUNT = 15
VAL_COUNT = 15

COLORS = "250 50 83 gryaz"
OUT_FOLDER = "out"

shutil.rmtree(OUT_FOLDER, ignore_errors=True)

subsets = ["Train", "Validation", "Test"]
img_folders = {}
mask_folders = {}

img_paths = glob(f"{IMGS_FOLDER}/*")
shuffle(img_paths)

img_paths = {
    "Train": img_paths[TEST_COUNT:-VAL_COUNT],
    "Validation": img_paths[-VAL_COUNT:],
    "Test": img_paths[:TEST_COUNT],
}


for subset in subsets:
    base_folder = f"{OUT_FOLDER}/{subset}_task"

    img_folder = f"{base_folder}/{subset}"
    img_folders[subset] = img_folder

    mask_folder = f"{base_folder}/{subset}annot"
    mask_folders[subset] = mask_folder

    Path(img_folder).mkdir(parents=True, exist_ok=True)
    Path(mask_folder).mkdir(parents=True, exist_ok=True)

    with open(f"{base_folder}/label_colors.txt", "w") as f:
        f.write(COLORS)

    with open(f"{base_folder}/{subset}.txt", "w") as f:
        for img_pth in tqdm(img_paths[subset]):
            mask_pth = img_pth.replace("open_img", "open_msk")[:-3] + "png"
            new_img_pth = f"{img_folder}/{basename(img_pth)}"
            new_mask_pth = f"{mask_folder}/{basename(mask_pth)}"

            shutil.copy(img_pth, new_img_pth)
            shutil.copy(mask_pth, new_mask_pth)

            f.write(f"/{subset}/{basename(img_pth)} {subset}annot/{basename(mask_pth)}\n")
