from glob import glob
from pathlib import Path
import shutil
from os.path import basename
import cv2
from tqdm import tqdm

"""
Превращает датасет, данный организаторами, в Imagenet формат
"""


OUT_FOLDER = "classification_dataset"
IMAGES_PATHS = {
    "Train": glob("out/Train_task/Train/*"),
    "Valid": glob("out/Test_task/Test/*") + glob("out/Validation_task/Validation/*"),
}
MASKS_PATHS = {
    "Train": glob("out/Train_task/Trainannot/*"),
    "Valid": glob("out/Test_task/Testannot/*") + glob("out/Validation_task/Validationannot/*"),
}

THRESH_DIRTY = 0.9
THRESH_CLEAN = 0.01

shutil.rmtree(OUT_FOLDER, ignore_errors=True)

for subset in IMAGES_PATHS.keys():
    for img_pth, mask_pth in tqdm(zip(IMAGES_PATHS[subset], MASKS_PATHS[subset])):
        mask = cv2.imread(mask_pth, cv2.IMREAD_GRAYSCALE) > 0
        norm_square = mask.sum() / mask.size
        if norm_square > THRESH_DIRTY:
            if norm_square < 1:
                print(norm_square)
            cls = "dirty"
        elif norm_square < THRESH_CLEAN:
            cls = "clean"
        else:
            cls = "suspected"

        new_subset = "val" if subset == "Valid" else "train"

        base_folder = f"{OUT_FOLDER}/{new_subset}/{cls}"
        Path(base_folder).mkdir(parents=True, exist_ok=True)

        new_img_pth = f"{base_folder}/{basename(img_pth)}"

        shutil.copy(img_pth, new_img_pth)
