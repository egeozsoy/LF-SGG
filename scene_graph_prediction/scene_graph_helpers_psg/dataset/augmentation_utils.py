from typing import List, Tuple, Optional, Dict

import math
import torch
from torch import Tensor
from torchvision.transforms import RandomResizedCrop
from torchvision.transforms import functional as F, InterpolationMode


def _apply_op(img: Tensor, op_name: str, magnitude: float,
              interpolation: InterpolationMode, fill: Optional[List[float]]):
    if op_name == "ShearX":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[math.degrees(magnitude), 0.0],
                       interpolation=interpolation, fill=fill)
    elif op_name == "ShearY":
        img = F.affine(img, angle=0.0, translate=[0, 0], scale=1.0, shear=[0.0, math.degrees(magnitude)],
                       interpolation=interpolation, fill=fill)
    elif op_name == "TranslateX":
        img = F.affine(img, angle=0.0, translate=[int(magnitude), 0], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "TranslateY":
        img = F.affine(img, angle=0.0, translate=[0, int(magnitude)], scale=1.0,
                       interpolation=interpolation, shear=[0.0, 0.0], fill=fill)
    elif op_name == "Rotate":
        img = F.rotate(img, magnitude, interpolation=interpolation, fill=fill)
    elif op_name == "Brightness":
        img = F.adjust_brightness(img, 1.0 + magnitude)
    elif op_name == "Color":
        img = F.adjust_saturation(img, 1.0 + magnitude)
    elif op_name == "Contrast":
        img = F.adjust_contrast(img, 1.0 + magnitude)
    elif op_name == "Sharpness":
        img = F.adjust_sharpness(img, 1.0 + magnitude)
    elif op_name == "Posterize":
        img = F.posterize(img, int(magnitude))
    elif op_name == "Solarize":
        img = F.solarize(img, magnitude)
    elif op_name == "AutoContrast":
        img = F.autocontrast(img)
    elif op_name == "Equalize":
        img = F.equalize(img)
    elif op_name == "Invert":
        img = F.invert(img)
    elif op_name == "Crop":
        img = RandomResizedCrop(size=(img.size[1], img.size[0]), scale=(magnitude, magnitude))(img)
    elif op_name == "Identity":
        pass
    else:
        raise ValueError("The provided operator {} is not recognized.".format(op_name))
    return img


class TrivialAugmentWide(torch.nn.Module):
    r"""Dataset-independent data-augmentation with TrivialAugment Wide, as described in
    `"TrivialAugment: Tuning-free Yet State-of-the-Art Data Augmentation" <https://arxiv.org/abs/2103.10158>`.
    If the image is torch Tensor, it should be of type torch.uint8, and it is expected
    to have [..., 1 or 3, H, W] shape, where ... means an arbitrary number of leading dimensions.
    If img is PIL Image, it is expected to be in mode "L" or "RGB".
    Args:
        num_magnitude_bins (int): The number of different magnitude values.
        interpolation (InterpolationMode): Desired interpolation enum defined by
            :class:`torchvision.transforms.InterpolationMode`. Default is ``InterpolationMode.NEAREST``.
            If input is Tensor, only ``InterpolationMode.NEAREST``, ``InterpolationMode.BILINEAR`` are supported.
        fill (sequence or number, optional): Pixel fill value for the area outside the transformed
            image. If given a number, the value is used for all bands respectively.
        """

    def __init__(self, num_magnitude_bins: int = 31, interpolation: InterpolationMode = InterpolationMode.NEAREST,
                 fill: Optional[List[float]] = None) -> None:
        super().__init__()
        self.num_magnitude_bins = num_magnitude_bins
        self.interpolation = interpolation
        self.fill = fill

    def _augmentation_space(self, num_bins: int) -> Dict[str, Tuple[Tensor, bool]]:
        # return {
        #     # op_name: (magnitudes, signed)
        #     "Identity": (torch.tensor(0.0), False),
        #     "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
        #     "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
        #     "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
        #     "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "Color": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
        #     "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
        #     "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
        #     "AutoContrast": (torch.tensor(0.0), False),
        #     "Equalize": (torch.tensor(0.0), False),
        # }
        return {
            # op_name: (magnitudes, signed)
            # "Identity": (torch.tensor(0.0), False),
            "ShearX": (torch.linspace(0.0, 0.99, num_bins), True),
            "ShearY": (torch.linspace(0.0, 0.99, num_bins), True),
            "TranslateX": (torch.linspace(0.0, 32.0, num_bins), True),
            "TranslateY": (torch.linspace(0.0, 32.0, num_bins), True),
            # "Rotate": (torch.linspace(0.0, 135.0, num_bins), True),
            "Rotate": (torch.linspace(0.0, 60, num_bins), True),
            "Brightness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Color": (torch.linspace(0.0, 0.99, num_bins), True),
            # "Contrast": (torch.linspace(0.0, 0.99, num_bins), True),
            # "Sharpness": (torch.linspace(0.0, 0.99, num_bins), True),
            "Posterize": (8 - (torch.arange(num_bins) / ((num_bins - 1) / 6)).round().int(), False),
            # "Solarize": (torch.linspace(256.0, 0.0, num_bins), False),
            # "AutoContrast": (torch.tensor(0.0), False),
            # "Equalize": (torch.tensor(0.0), False),
            "Crop": (torch.linspace(0.7, 1.0, num_bins), False),
        }

    def forward(self, img: Tensor) -> Tensor:
        """
            img (PIL Image or Tensor): Image to be transformed.
        Returns:
            PIL Image or Tensor: Transformed image.
        """
        fill = self.fill
        if isinstance(img, Tensor):
            if isinstance(fill, (int, float)):
                fill = [float(fill)] * F.get_image_num_channels(img)
            elif fill is not None:
                fill = [float(f) for f in fill]

        op_meta = self._augmentation_space(self.num_magnitude_bins)
        op_index = int(torch.randint(len(op_meta), (1,)).item())
        op_name = list(op_meta.keys())[op_index]
        magnitudes, signed = op_meta[op_name]
        magnitude = float(magnitudes[torch.randint(len(magnitudes), (1,), dtype=torch.long)].item()) \
            if magnitudes.ndim > 0 else 0.0
        if signed and torch.randint(2, (1,)):
            magnitude *= -1.0

        return _apply_op(img, op_name, magnitude, interpolation=self.interpolation, fill=fill)

    def __repr__(self) -> str:
        s = self.__class__.__name__ + '('
        s += 'num_magnitude_bins={num_magnitude_bins}'
        s += ', interpolation={interpolation}'
        s += ', fill={fill}'
        s += ')'
        return s.format(**self.__dict__)


def get_crop_image_augmentations(config, split, for_eval):
    # define image augmentations occording to self.config
    if config['dataset']['data_augmentation'] and split == 'train' and not for_eval:
        return TrivialAugmentWide()
    else:
        return None
