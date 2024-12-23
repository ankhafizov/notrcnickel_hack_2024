from dataclasses import dataclass
import numpy as np
import pandas as pd


@dataclass
class FrameElement:
    image: np.ndarray
    mask: np.ndarray
    mask_over_img: np.ndarray
    filename: str
    cls: str
    characteristics: pd.DataFrame
