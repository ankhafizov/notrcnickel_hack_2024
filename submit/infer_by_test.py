import os
import sys
import json
import base64
import cv2
import numpy as np
from ultralytics import YOLO

dataset_path, output_path = sys.argv[1:]
classifier_model_path = "./classifier.pt"
model_path = "./segmentation.pt"

segment_model = YOLO(model_path)
segment_model_img_size = 448
segment_model_conf = 0.75
segment_model.to("cpu")

classifier_model = YOLO(classifier_model_path)
classifier_model_img_size = 640
classifier_model.to("cpu")


def infer_segmentation(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    result = segment_model.predict(
        image, imgsz=segment_model_img_size, verbose=False, conf=segment_model_conf
    )[0]
    mask = np.zeros((height, width), dtype=np.uint8)

    masks = result.masks
    if masks is not None:
        for mask_array in masks.data.cpu().numpy():
            mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_LINEAR)
            mask[mask_array > 0] = 255

    return mask


def infer_classifier(img_path):
    img = cv2.imread(img_path)
    result = classifier_model.predict(img, imgsz=classifier_model_img_size, verbose=False)[0]
    probs = result.probs.data.cpu().numpy()
    return result.names[probs.argmax()]


def create_full_mask(image_path, labeled=True):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    mask = (
        np.full((height, width), 255, dtype=np.uint8)
        if labeled
        else np.zeros((height, width), dtype=np.uint8)
    )
    return mask


results_dict = {}
for image_name in os.listdir(dataset_path):
    if image_name.lower().endswith(".jpg"):
        img_path = os.path.join(dataset_path, image_name)
        cls = infer_classifier(img_path)
        if cls == "suspected":
            bin_mask = infer_segmentation(img_path)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
            opened = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
            mask = opened
        elif cls == "clean":
            mask = create_full_mask(img_path, False)
        elif cls == "dirty":
            mask = create_full_mask(img_path, True)

        _, encoded_img = cv2.imencode(".png", mask)

        encoded_str = base64.b64encode(encoded_img).decode("utf-8")
        results_dict[image_name] = encoded_str

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(results_dict, f, ensure_ascii=False)
