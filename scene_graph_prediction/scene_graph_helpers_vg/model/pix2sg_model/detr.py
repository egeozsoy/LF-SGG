# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import json
from argparse import Namespace
from collections import defaultdict

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.backbone import build_backbone
from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.detr_configs import NO_KNOWN_TOKEN, MAX_NUM_RELS, RANDOM_MAX_INSTANCE_ID, \
    INSTANCE_START, START_TOKEN, NOISE_TOKEN, END_TOKEN, ASSUMED_MAX_INSTANCE_ID, PRED_START, ENTITY_START, N_PREDS, N_ENTITIES
from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.transformer import build_transformer
from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.util.misc import (NestedTensor, nested_tensor_from_tensor_list)


class DETR(nn.Module):
    """ This is the DETR module that performs object detection """

    def __init__(self, image_backbone, transformer, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.vocab_embed = MLP(hidden_dim, hidden_dim, NO_KNOWN_TOKEN + 1, 3)  # Allow #NO_KNOWN_TOKEN+1 tokens
        if image_backbone is not None:
            self.input_proj_image = nn.Conv2d(image_backbone.num_channels, hidden_dim, kernel_size=1)
        self.image_backbone = image_backbone

    def forward(self, image_samples: NestedTensor, sequence, max_num_rels=None):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        image_src, image_mask, image_pos_embed, point_src, point_mask, point_pos_embed = None, None, None, None, None, None
        if isinstance(image_samples, (list, torch.Tensor)):
            image_samples = nested_tensor_from_tensor_list(image_samples)
        image_features, image_pos = self.image_backbone(image_samples)

        image_src, image_mask = image_features[-1].decompose()
        image_src = self.input_proj_image(image_src)
        image_pos_embed = image_pos[-1]
        # We are not working with batches here, but everything should be seen at once
        assert image_mask is not None

        hs = self.transformer(image_src, image_mask, image_pos_embed, sequence, self.vocab_embed, max_num_rels=max_num_rels)
        if self.training:
            out = self.vocab_embed(hs[0])
        else:
            out = hs
        return out


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def preprocess_input(batch, device, for_train=True):
    '''
    0-50: pred_indices (51 preds), 51:-201(151 rels):  250 - 750: bboxes coordinates.
    :param batch:
    :param device:
    :param for_train:
    :param bins:
    :return:
    '''

    if for_train:
        all_input_seqs = []
        all_rel_label = []
        for elem in batch[1]:
            rel_label = torch.tensor(elem['relation']).long().to(device)
            # change format to sub, sub_id, obj, obj_id, rel
            rel_label = rel_label[:, [0, 1, 3, 4, 2]]
            # We can expect at most MAX_NUM_RELS rels, so cut after MAX_NUM_RELS
            if len(rel_label) > MAX_NUM_RELS:
                rel_label = rel_label[:MAX_NUM_RELS]
                # print(f'Reducing Rels to {MAX_NUM_RELS}')

            random_sub_ids = torch.randint(51, 203, (MAX_NUM_RELS - len(rel_label),)).long().cpu().unsqueeze(1)
            random_obj_ids = torch.randint(51, 203, (MAX_NUM_RELS - len(rel_label),)).long().cpu().unsqueeze(1)
            random_pred_ids = torch.randint(0, 51, (MAX_NUM_RELS - len(rel_label),)).long().cpu().unsqueeze(1)
            random_sub_instances = (torch.rand(MAX_NUM_RELS - len(rel_label), 1).cpu() * RANDOM_MAX_INSTANCE_ID).int() + INSTANCE_START
            random_obj_instances = (torch.rand(MAX_NUM_RELS - len(rel_label), 1).cpu() * RANDOM_MAX_INSTANCE_ID).int() + INSTANCE_START
            # change format to sub, sub_id, obj, obj_id, rel
            random_rel_label = torch.cat([random_sub_ids, random_sub_instances, random_obj_ids, random_obj_instances, random_pred_ids], dim=1).long().to(device)

            input_seqs = torch.cat([rel_label, random_rel_label], dim=0)
            input_seqs = torch.cat([torch.ones(1, device=rel_label.device, dtype=rel_label.dtype) * START_TOKEN, input_seqs.flatten()])

            # sum of the following channels should equal triplet representation size
            output_na = torch.ones(MAX_NUM_RELS - len(rel_label), 3, device=rel_label.device, dtype=rel_label.dtype) * NO_KNOWN_TOKEN

            output_noise = torch.ones(MAX_NUM_RELS - len(rel_label), 1, device=rel_label.device, dtype=rel_label.dtype) * NOISE_TOKEN
            output_end = torch.ones(MAX_NUM_RELS - len(rel_label), 1, device=rel_label.device, dtype=rel_label.dtype) * END_TOKEN

            output_seqs = torch.cat([output_na, output_noise, output_end], dim=-1)
            output_seqs = torch.cat([rel_label.flatten(), torch.ones(1, device=rel_label.device, dtype=rel_label.dtype) * END_TOKEN, output_seqs.flatten()])

            input_seqs = input_seqs.unsqueeze(0)
            output_seqs = output_seqs.unsqueeze(0)
            rel_label = output_seqs.flatten()

            all_input_seqs.append(input_seqs)
            all_rel_label.append(rel_label)
        return torch.cat(all_input_seqs), torch.cat(all_rel_label)

    else:
        all_input_seqs = []
        all_rel_label = []
        for elem in batch[1]:
            rel_label = torch.tensor(elem['relation']).long().to(device)
            input_seq = torch.ones(1, 1, device=device, dtype=torch.long) * START_TOKEN
            all_input_seqs.append(input_seq)
            all_rel_label.append(rel_label)
        return torch.cat(all_input_seqs), all_rel_label


def postprocess_output(model_output, targets, gts=None):
    all_outputs, all_values = model_output
    all_pred_rels = []
    all_pred_scores = []
    for output, values, target in zip(all_outputs, all_values, targets):
        output = output[1:].reshape(-1, 5)
        sub_id = output[:, 0].unsqueeze(-1).clip(ENTITY_START, ENTITY_START + N_ENTITIES - 1)
        sub_instance = (output[:, 1] - INSTANCE_START).unsqueeze(-1).clip(0, ASSUMED_MAX_INSTANCE_ID).int()
        obj_id = output[:, 2].unsqueeze(-1).clip(ENTITY_START, ENTITY_START + N_ENTITIES - 1)
        obj_instance = (output[:, 3] - INSTANCE_START).unsqueeze(-1).clip(0, ASSUMED_MAX_INSTANCE_ID).int()
        pred_id = output[:, 4].unsqueeze(-1).clip(PRED_START, PRED_START + N_PREDS - 1)

        values = values.reshape(-1, 5)
        sub_score = values[:, 0].unsqueeze(-1)
        obj_score = values[:, 2].unsqueeze(-1)
        pred_score = values[:, 4].unsqueeze(-1)

        all_pred_rels.append(torch.cat([sub_id, sub_instance, pred_id, obj_id, obj_instance], dim=-1))
        all_pred_scores.append(torch.cat([sub_score, pred_score, obj_score], dim=-1))

    if gts is not None:
        all_gt_rels = []
        for gt, target in zip(gts, targets):
            sub_id = gt[:, 0].unsqueeze(-1)
            sub_instance = (gt[:, 1] - INSTANCE_START).unsqueeze(-1).int()
            pred_id = gt[:, 2].unsqueeze(-1)
            obj_id = gt[:, 3].unsqueeze(-1)
            obj_instance = (gt[:, 4] - INSTANCE_START).unsqueeze(-1).int()

            all_gt_rels.append(torch.cat([sub_id, sub_instance, pred_id, obj_id, obj_instance], dim=-1))

        return all_pred_rels, all_pred_scores, all_gt_rels

    return all_pred_rels, all_pred_scores


def _apply_mapping_to_predictions(predictions, mapping):
    mapped_preds = []
    for sub_id, sub_instance, predicate_id, obj_id, obj_instance in predictions:
        if (sub_id, sub_instance) in mapping:
            sub_instance = mapping[(sub_id, sub_instance)]
        else:
            sub_instance = None
        if (obj_id, obj_instance) in mapping:
            obj_instance = mapping[(obj_id, obj_instance)]
        else:
            obj_instance = None
        mapped_preds.append((sub_id, sub_instance, predicate_id, obj_id, obj_instance))
    return mapped_preds


def _calculate_global_recall(gts, preds):
    gts_dict = {}
    for gt in gts:
        gt = tuple(gt)
        if gt not in gts_dict:
            gts_dict[gt] = 1
        else:
            gts_dict[gt] += 1

    preds_dict = {}
    for pred in preds:
        pred = tuple(pred)
        if pred not in preds_dict:
            preds_dict[pred] = 1
        else:
            preds_dict[pred] += 1

    correct_matches = 0.
    for gt_rel in gts_dict:
        if gt_rel in preds_dict:
            correct_matches += min(gts_dict[gt_rel], preds_dict[gt_rel])

    recall = correct_matches / len(gts)

    return recall


def _mean_recall_helper(gts, preds):
    '''
    Take note of the results per Ground truth Predicate category. We will mean them after every epoch to report mean recall
    '''
    gts_dict = {}
    for gt in gts:
        gt = tuple(gt)
        if gt not in gts_dict:
            gts_dict[gt] = 1
        else:
            gts_dict[gt] += 1

    preds_dict = {}
    for pred in preds:
        pred = tuple(pred)
        if pred not in preds_dict:
            preds_dict[pred] = 1
        else:
            preds_dict[pred] += 1

    per_predicate_recall = defaultdict(list)
    gt_predicate_count = defaultdict(int)
    for gt_rel in gts_dict:
        gt_predicate_count[gt_rel[2]] += gts_dict[gt_rel]
        if gt_rel in preds_dict:
            per_predicate_recall[gt_rel[2]].append(min(gts_dict[gt_rel], preds_dict[gt_rel]))
        else:
            per_predicate_recall[gt_rel[2]].append(0)

    # Calculate recall per predicate
    # We employ a trick: To make sure global weighting of recall is correct at the end, we add multiple copies of the recall
    for pred in per_predicate_recall:
        per_predicate_recall[pred] = [sum(per_predicate_recall[pred]) / gt_predicate_count[pred]] * gt_predicate_count[pred]

    return per_predicate_recall


def build(config):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    '''Namespace(aux_loss=True, backbone='resnet50', batch_size=4, bbox_loss_coef=5, clip_max_norm=0.1, coco_panoptic_path=None, coco_path='./coco/', dataset_file='coco', dec_layers=6, device='cpu', dice_loss_coef=1, dilation=False, dim_feedforward=1024, dist_url='env://', distributed=False, dropout=0.1, enc_layers=6, eos_coef=0.1, epochs=300, eval=False, frozen_weights=None, giou_loss_coef=2, hidden_dim=256, lr=0.0005, lr_backbone=1e-05, lr_drop=200, mask_loss_coef=1, masks=False, nheads=8, num_queries=100, num_workers=0, output_dir='./output/', position_embedding='sine', pre_norm=False, remove_difficult=False, resume='', seed=42, set_cost_bbox=5, set_cost_class=1, set_cost_giou=2, start_epoch=0, weight_decay=0.0001, world_size=1)'''
    model_args = {'aux_loss': True,
                  'image_backbone': config['MODEL']['IMAGE_MODEL'] if config['IMAGE_INPUT'] else None,
                  'point_backbone': True if config['PC_INPUT'] else None,
                  'batch_size': 4,
                  'bbox_loss_coef': 5,
                  'clip_max_norm': 0.1,
                  'coco_panoptic_path': None,
                  'coco_path': './coco/',
                  'dataset_file': 'coco',
                  'dec_layers': config['MODEL']['N_DECODERS'],
                  'device': 'cpu',
                  'dice_loss_coef': 1,
                  'dilation': False,
                  'dim_feedforward': 1024,
                  'dist_url': 'env://',
                  'distributed': False,
                  'dropout': 0.1,
                  'enc_layers': 6,
                  'eos_coef': 0.1,
                  'epochs': 300,
                  'eval': False,
                  'frozen_weights': None,
                  'giou_loss_coef': 2,
                  'hidden_dim': 256,
                  'lr': 0.0005,
                  'lr_backbone': 1e-05,
                  'lr_drop': 200,
                  'mask_loss_coef': 1,
                  'masks': False,
                  'nheads': 8,
                  'num_queries': 100,
                  'num_workers': 0,
                  'output_dir': './output/',
                  'position_embedding': 'sine',
                  'pre_norm': False,
                  'remove_difficult': False,
                  'resume': '',
                  'seed': 42,
                  'set_cost_bbox': 5,
                  'set_cost_class': 1,
                  'set_cost_giou': 2,
                  'start_epoch': 0,
                  'weight_decay': 0.0001,
                  'world_size': 1,
                  'freeze_first_N': config['MODEL']['FREEZE_FIRST_N']
                  }
    # Convert model args from dict to class
    model_args = Namespace(**model_args)
    pl.seed_everything(config['SEED'], workers=True)
    image_backbone = build_backbone(model_args)
    pl.seed_everything(config['SEED'], workers=True)
    transformer = build_transformer(model_args)
    pl.seed_everything(config['SEED'], workers=True)
    model = DETR(
        image_backbone,
        transformer,
        num_queries=model_args.num_queries,
        aux_loss=model_args.aux_loss,
    )
    with open('scene_graph_prediction/scene_graph_helpers_vg/dataset/pix2sg_pred_frequencies.json') as f:
        token_frequencies = json.load(f)
    # token_weights_dict = {k: 1 / v for k, v in token_frequencies.items()}  # linear weighting
    token_weights_dict = {k: 1 / (np.log(v) + 1) for k, v in token_frequencies.items()}  # log weighting, seems to work better in this case
    token_weights = torch.ones(NO_KNOWN_TOKEN + 1)
    # Update predicate weights for optimizing mean recall
    if config['MODEL']['WEIGHT_PREDICATES']:
        for k, v in token_weights_dict.items():
            token_weights[int(k)] = v
        min_weight = min(token_weights_dict.values())
        # first token is __background__ predicate, adjust that as well
        token_weights[0] = min_weight
        extra_token_weight = min_weight / 100
        token_weights[END_TOKEN] = extra_token_weight
        token_weights[NOISE_TOKEN] = extra_token_weight
    else:
        token_weights[END_TOKEN] = 0.01
        token_weights[NOISE_TOKEN] = 0.01
    criterion = torch.nn.CrossEntropyLoss(weight=token_weights)

    return model, criterion, preprocess_input, postprocess_output
