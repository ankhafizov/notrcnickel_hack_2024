import cv2
import numpy as np
from ultralytics import YOLO
from elements.FrameElement import FrameElement
import pandas as pd
from os.path import basename


class AIHelper:
    def __init__(self, config):
        self.common_config = config["common"]

        self.classification_config = config["classification"]
        self.segmentation_config = config["segmentation"]

        self.classification_model = YOLO(self.classification_config["weights"])
        self.segmentation_model = YOLO(self.segmentation_config["weights"])

    def predict(self, img_path):
        img = cv2.imread(img_path)
        img_h, img_w, _ = img.shape
        cls = self._infer_yolo_classifier(img)

        if cls == "clean":
            mask = np.zeros((img_h, img_w))
        elif cls == "dirty":
            mask = np.ones((img_h, img_w))
        else:
            mask = self._infer_yolo_segmenter(img)

        dirty_degree = round((mask > 0).sum() / mask.size, 2)
        characteristics = pd.DataFrame(
            [
                ["Загрязненность, %", dirty_degree],
                ["Площадь маски, Мп", round(img.size / 1_000_000, 2)],
            ],
            columns=["Характеристика", "Значение"],
        )

        cls = (
            self.common_config["dirty_label"]
            if dirty_degree > 0
            else self.common_config["clean_label"]
        )
        print(cls)
        mask_over_img = self._get_image_with_mask(img, mask)

        frame_element = FrameElement(
            img, mask, mask_over_img, basename(img_path), cls, characteristics
        )

        return frame_element

    def _infer_yolo_classifier(self, img):
        result = self.classification_model.predict(
            img, imgsz=self.classification_config["imgsz"], verbose=False
        )[0]

        probs = result.probs.data.cpu().numpy()
        cls = result.names[probs.argmax()]
        return cls

    def _infer_yolo_segmenter(self, img):
        height, width = img.shape[:2]
        result = self.segmentation_model.predict(
            img, imgsz=self.segmentation_config["imgsz"], verbose=False
        )[0]
        mask = np.zeros((height, width), dtype=np.uint8)

        masks = result.masks
        if masks is not None:
            for mask_array in masks.data.cpu().numpy():
                mask_array = cv2.resize(mask_array, (width, height), interpolation=cv2.INTER_LINEAR)
                mask[mask_array > 0] = 255

        return mask

    def _get_image_with_mask(self, img, mask, alpha=0.4):
        print(mask.shape)
        h, w = mask.shape
        red_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        red_overlay[:, :, 0] = (mask > 0).astype(np.uint8) * 255

        overlaid_image = cv2.addWeighted(img, 1.0, red_overlay, alpha, 0)

        return overlaid_image
