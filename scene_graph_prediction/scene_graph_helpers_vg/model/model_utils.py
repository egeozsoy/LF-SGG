from itertools import chain

import timm
from timm.data import resolve_data_config, create_transform
from transformers import DetrFeatureExtractor, DetrForSegmentation


def get_image_model(model_config, only_transforms=False):
    if model_config['IMAGE_MODEL'] == False:
        return None
    if 'detr-resnet' in model_config['IMAGE_MODEL']:
        feature_extractor = DetrFeatureExtractor.from_pretrained(f'facebook/{model_config["IMAGE_MODEL"]}')
        model = DetrForSegmentation.from_pretrained(f'facebook/{model_config["IMAGE_MODEL"]}')
        image_transformations = {'train': feature_extractor, 'val': feature_extractor, 'test': feature_extractor}
        if only_transforms:
            return image_transformations

        return model, image_transformations
    else:
        # Default: timm==0.4.12, but for eva: timm==0.9.5
        image_model = timm.create_model(model_config['IMAGE_MODEL'], pretrained=True, num_classes=0)
        # Freeze the whole model
        for param in image_model.parameters():
            param.requires_grad = False
        # Unfreeze conv head
        if hasattr(image_model, 'conv_head'):
            for param in chain(image_model.conv_head.parameters()):
                param.requires_grad = True
        image_config = resolve_data_config({}, model=image_model)
        val_image_transform = create_transform(**image_config)
        image_transformations = {'train': val_image_transform, 'val': val_image_transform, 'test': val_image_transform}
        if only_transforms:
            return image_transformations
        return image_model, image_transformations
