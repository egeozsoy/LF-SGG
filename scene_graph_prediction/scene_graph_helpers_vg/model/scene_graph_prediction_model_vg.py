# !/usr/bin/env python3
# -*- coding: utf-8 -*-
from datetime import datetime

import cv2
import numpy as np
from matplotlib import pyplot as plt
from torch.distributed import all_gather_object
from torch.utils.data import DataLoader
from tqdm import tqdm

from scene_graph_prediction.branched_ssg_matcher import PyBranchedSSGMatcher as BranchedSSGMatcher

if __name__ == '__main__' and __package__ is None:
    from os import sys

    sys.path.append('../')
import torch
from torch import nn
import torch.optim as optim
import pytorch_lightning as pl
from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.detr import build as build_pix2sg_model, NO_KNOWN_TOKEN, \
    _apply_mapping_to_predictions, _calculate_global_recall, _mean_recall_helper, ENTITY_START, PRED_START
from copy import deepcopy


class SGPNModelWrapperVG(pl.LightningModule):
    def __init__(self, config, ind_to_classes, ind_to_predicates):
        super().__init__()
        self.config = config
        self.mconfig = config['MODEL']
        self.ind_to_classes, self.ind_to_predicates = ind_to_classes, ind_to_predicates
        self.lr = float(self.config['LR'])
        # evaluation metrics
        self.train_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}  # mean
        self.val_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}
        self.reset_metrics()

        self.pix2sg_model, self.pix2sg_criterion, self.pix2sg_preprocessor, self.pix2sg_postprocessor = build_pix2sg_model(self.config)

    def freeze_image_model_batchnorm(self):
        models_to_freeze = []
        if self.config['IMAGE_INPUT'] == 'full':
            models_to_freeze.append(self.full_image_model)
        if self.config['IMAGE_INPUT'] == 'crop':
            models_to_freeze.append(self.obj_encoder)
            models_to_freeze.append(self.rel_encoder)

        for image_model in models_to_freeze:
            for module in image_model.modules():
                if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
                    if hasattr(module, 'weight'):
                        module.weight.requires_grad_(False)
                    if hasattr(module, 'bias'):
                        module.bias.requires_grad_(False)
                    module.eval()

    def forward(self, batch, return_meta_data=False, pix2sg_training=True):
        if pix2sg_training:
            self.train()
            input_seqs, rel_label = self.pix2sg_preprocessor(batch, device=self.device, for_train=True)
            model_output = self.pix2sg_model(image_samples=batch[0], sequence=input_seqs)
            model_output = model_output[-1].reshape(-1, NO_KNOWN_TOKEN + 1)
            model_output_known = model_output[rel_label != NO_KNOWN_TOKEN]
            rel_label_known = rel_label[rel_label != NO_KNOWN_TOKEN]
            return model_output_known, rel_label_known
        else:
            self.eval()
            input_seq, rel_label = self.pix2sg_preprocessor(batch, device=self.device, for_train=False)
            image_samples = torch.cat([elem['full_image'] for elem in batch]).to(self.device) if self.config['IMAGE_INPUT'] else None
            point_samples = torch.stack([elem['points'] for elem in batch]).transpose(1, 2).to(self.device) if self.config['PC_INPUT'] else None
            model_output = self.pix2sg_model(image_samples=image_samples, point_samples=point_samples, sequence=input_seq)
            model_output, attn_maps = model_output
            result = self.pix2sg_postprocessor(model_output)
            return result, rel_label

    def reset_metrics(self, split=None):
        if split == 'train':
            self.train_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}
        elif split == 'val':
            self.val_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}
        else:
            self.train_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}
            self.val_recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}

    def training_step(self, batch, batch_idx):
        model_output, rel_labels = self(batch, return_meta_data=True, pix2sg_training=True)
        loss = self.pix2sg_criterion(model_output, rel_labels)

        if batch_idx % 100 == 0:
            print(f'{datetime.now()}: Training step: {batch_idx} ---- Loss: {loss.item()}')

        return loss

    def validation_step(self, batch, batch_idx):
        model_output, rel_labels = self(batch, return_meta_data=True, pix2sg_training=True)
        loss = self.pix2sg_criterion(model_output, rel_labels)

        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input_seq, _ = self.pix2sg_preprocessor(batch, device=self.device, for_train=False)
        # batch[0] = batch[0].to(self.device)
        model_output = self.pix2sg_model(image_samples=batch[0], sequence=input_seq, max_num_rels=100)
        model_output = model_output[:-1]

        all_pred_rels, _ = self.pix2sg_postprocessor(model_output, targets=batch[1])
        all_human_readable_pred_triplets = []
        all_elems = []
        for overall_idx, (pred_triplets, elem) in enumerate(zip(all_pred_rels, batch[1])):
            pred_triplets = pred_triplets.cpu().tolist()
            human_readable_pred_triplets = self.convert_predictions_to_human_readable(pred_triplets)
            all_human_readable_pred_triplets.append(human_readable_pred_triplets)
            all_elems.append(elem)

        return (all_elems, all_human_readable_pred_triplets)

    # def test_step(self, batch, batch_idx): # not for inference
    #     return self.validation_step(batch, batch_idx)

    def plot_attn_map_on_image(self, image, attn_map, name):
        image = image.cpu().numpy().transpose(1, 2, 0)
        image = (image * [0.229, 0.224, 0.225]) + [0.485, 0.456, 0.4060]
        plt.imshow(image)
        # clean image plot, without axis and ticks or white space, just the image
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0)
        plt.savefig(f'{name}___image.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

        attn_map = attn_map.cpu().numpy().reshape(19, 19)
        # scale attention map to image size
        attn_map = cv2.resize(attn_map, (image.shape[1], image.shape[0]))
        # normalize attention map
        attn_map = (attn_map - np.min(attn_map)) / (np.max(attn_map) - np.min(attn_map))
        plt.imshow(attn_map, cmap='jet')
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0)
        plt.savefig(f'{name}___attn_map.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

        # plot the image
        plt.imshow(image)
        # plot the attention map
        plt.imshow(attn_map, cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout(pad=0)
        plt.savefig(f'{name}___attn_map_on_image.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()

    def convert_predictions_to_human_readable(self, pred_triplets):
        '''
        pred_triplets: [sub_id, sub_instance, pred_id, obj_id, obj_instance]
        '''
        human_readable_triplets = []
        for triplet in pred_triplets:
            sub_id, sub_instance, pred_id, obj_id, obj_instance = triplet
            sub_name = self.ind_to_classes[sub_id - ENTITY_START]
            pred_name = self.ind_to_predicates[pred_id - PRED_START]
            obj_name = self.ind_to_classes[obj_id - ENTITY_START]
            human_readable_triplets.append((sub_name, sub_instance, pred_name, obj_name, obj_instance))
        return human_readable_triplets

    @torch.no_grad()
    def evaluate_pix2sg(self, dataset, max_num_rels=100):
        dataloader = DataLoader(dataset, batch_size=self.config['BATCH_SIZE'], shuffle=False, num_workers=self.config['NUM_WORKERS'], pin_memory=True, collate_fn=dataset.collate_fn)

        model, _, preprocessor, postprocessor = build_pix2sg_model(self.config)  # Create new model
        model.load_state_dict(self.pix2sg_model.state_dict())
        model.eval()
        model.to(self.device)
        branched_ssg_matcher = BranchedSSGMatcher()

        recalls = {20: [], 50: [], 100: [], 'Triplet': [], 'Entity': [], 'Pred': [], 'm20': {}, 'm50': {}, 'm100': {}}
        for batch_idx, batch in enumerate(tqdm(dataloader, desc='Evaluating pix2sg')):
            input_seq, all_rel_labels = preprocessor(batch, device=self.device, for_train=False)
            batch[0] = batch[0].to(self.device)
            model_output = model(image_samples=batch[0], sequence=input_seq, max_num_rels=max_num_rels)
            model_output, all_attn_maps = model_output[:-1], model_output[-1]
            all_attn_maps = torch.stack(all_attn_maps).transpose(0, 1)

            all_pred_rels, all_pred_scores, all_gt_rels = postprocessor(model_output, targets=batch[1], gts=all_rel_labels)
            for overall_idx, (pred_triplets, pred_triplet_scores, gt_triplets, attn_maps, elem) in enumerate(
                    zip(all_pred_rels, all_pred_scores, all_gt_rels, all_attn_maps, batch[1])):
                pred_triplets = pred_triplets.cpu().tolist()
                pred_triplet_scores = pred_triplet_scores.cpu().tolist()
                gt_triplets = gt_triplets.cpu().tolist()
                human_readable_gt_triplets = sorted(self.convert_predictions_to_human_readable(gt_triplets))

                for threshold in recalls.keys():
                    if threshold == 'Triplet':  # this is not a threshold, but a flag to compute triplet recall
                        gt_triplet_set = {(g[0], g[2], g[3]) for g in gt_triplets}
                        pred_triplet_set = {(p[0], p[2], p[3]) for p in pred_triplets}
                        recalls[threshold].append(len(gt_triplet_set.intersection(pred_triplet_set)) / len(gt_triplet_set))
                    elif threshold == 'Entity':
                        gt_entity_set = {(g[0], g[3]) for g in gt_triplets}
                        pred_entity_set = {(p[0], p[3]) for p in pred_triplets}
                        recalls[threshold].append(len(gt_entity_set.intersection(pred_entity_set)) / len(gt_entity_set))
                    elif threshold == 'Pred':
                        gt_pred_set = {g[2] for g in gt_triplets}
                        pred_pred_set = {p[2] for p in pred_triplets}
                        recalls[threshold].append(len(gt_pred_set.intersection(pred_pred_set)) / len(gt_pred_set))
                    elif threshold == 20 or threshold == 50 or threshold == 100:  # skip the m20, m50, m100 (they are computer here)
                        # postprocessor already sorted the predictions by score
                        # remove duplicates
                        pred_triplets_k = []
                        for p in pred_triplets:
                            p = tuple(p)
                            if p not in pred_triplets_k:
                                pred_triplets_k.append(p)
                        # If number of unique enough use it, otherwise just use all
                        if len(pred_triplets_k) >= threshold:
                            pred_triplets_k = pred_triplets_k[:threshold]
                        else:
                            pred_triplets_k = pred_triplets[:threshold]
                        pred_triplets_k = pred_triplets_k[:threshold]
                        instance_id_mapping_branched = branched_ssg_matcher.branched_matching(deepcopy(gt_triplets), deepcopy(pred_triplets_k), N=3, depth_limit=10)
                        mapped_pred_triplets = _apply_mapping_to_predictions(pred_triplets_k, instance_id_mapping_branched)
                        human_readable_pred_triplets = self.convert_predictions_to_human_readable(mapped_pred_triplets)
                        recall = _calculate_global_recall(gt_triplets, mapped_pred_triplets)
                        recalls[threshold].append(recall)
                        per_predicate_recall = _mean_recall_helper(gt_triplets, mapped_pred_triplets)
                        for predicate, pred_recall in per_predicate_recall.items():
                            if predicate not in recalls[f'm{threshold}']:
                                recalls[f'm{threshold}'][predicate] = []
                            recalls[f'm{threshold}'][predicate].extend(pred_recall)

        return recalls

    def log_recall(self, epoch_loss, split):
        def _calculate_mean_recall(mean_recalls_at_K: dict):
            '''
            Mean over predicate classes
            '''
            mean_recalls = []
            for recalls in mean_recalls_at_K.values():
                mean_recalls.append(np.mean(recalls))
            return np.mean(mean_recalls)

        if split == 'train':
            recalls = self.train_recalls
        elif split == 'val':
            recalls = self.val_recalls
        else:
            raise NotImplementedError()
        if epoch_loss is not None:
            self.log(f'Epoch_Loss/{split}', epoch_loss, rank_zero_only=True)

        self.log(f'Epoch_Recall_20/{split}', np.mean(recalls[20]), rank_zero_only=True)
        self.log(f'Epoch_Recall_50/{split}', np.mean(recalls[50]), rank_zero_only=True)
        self.log(f'Epoch_Recall_100/{split}', np.mean(recalls[100]), rank_zero_only=True)
        self.log(f'Epoch_Recall_Triplet/{split}', np.mean(recalls['Triplet']), rank_zero_only=True)
        self.log(f'Epoch_Recall_Entity/{split}', np.mean(recalls['Entity']), rank_zero_only=True)
        self.log(f'Epoch_Recall_Pred/{split}', np.mean(recalls['Pred']), rank_zero_only=True)
        mRecall_20 = _calculate_mean_recall(recalls['m20'])
        mRecall_50 = _calculate_mean_recall(recalls['m50'])
        mRecall_100 = _calculate_mean_recall(recalls['m100'])
        self.log(f'Epoch_mRecall_20/{split}', mRecall_20, rank_zero_only=True)
        self.log(f'Epoch_mRecall_50/{split}', mRecall_50, rank_zero_only=True)
        self.log(f'Epoch_mRecall_100/{split}', mRecall_100, rank_zero_only=True)

        # Print a good looking table
        print(f'Split: {split}\n'
              f'        Recall@20: {np.mean(recalls[20]):.4f} Recall@50: {np.mean(recalls[50]):.4f} Recall@100: {np.mean(recalls[100]):.4f}\n'
              f'        mRecall@20: {mRecall_20:.4f} mRecall@50: {mRecall_50:.4f} mRecall@100: {mRecall_100:.4f}\n'
              f'        Triplet: {np.mean(recalls["Triplet"]):.4f} Entity: {np.mean(recalls["Entity"]):.4f} Pred: {np.mean(recalls["Pred"]):.4f}')

    def on_validation_epoch_end(self):
        # log with current time
        print(f'{datetime.now()}: Validating Epoch End....')
        if self.global_rank == 0:  # Evaluation completely on one process, no need to sync. Check ddp_reference.py for distributed evaluation
            if self.trainer.train_dataloader is not None and self.current_epoch % 10 == 1:
                self.train_recalls = self.evaluate_pix2sg(self.trainer.train_dataloader.dataset,
                                                          max_num_rels=20)  # TODO 300 gives the best results but it really slow, 20 gives ok results and very fast
                self.log_recall(epoch_loss=None, split='train')
                self.reset_metrics(split='train')
            self.val_recalls = self.evaluate_pix2sg(self.trainer.val_dataloaders.dataset, max_num_rels=20)  # TODO 300 gives the best results but it really slow, 20 gives ok results and very fast
            self.log_recall(epoch_loss=None, split='val')
            self.reset_metrics(split='val')
        torch.distributed.barrier()

    def configure_optimizers(self):
        optimizer = optim.AdamW(params=self.parameters(), lr=self.lr, weight_decay=float(self.config['W_DECAY']))
        return optimizer
