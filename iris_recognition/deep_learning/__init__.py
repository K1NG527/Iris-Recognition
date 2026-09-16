"""Deep Learning Module for Iris Recognition."""

from .model import IrisDeepNet, ArcMarginProduct
from .dataset import NormalizedIrisDataset
from .matcher import DeepIrisMatcher

__all__ = ["IrisDeepNet", "ArcMarginProduct", "NormalizedIrisDataset", "DeepIrisMatcher"]
