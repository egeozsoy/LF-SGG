import json
import random
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from transformers import DetrFeatureExtractor

from helpers.psg_helpers.configurations import ANN_FILE_PATH, DATA_ROOT_PATH
from scene_graph_prediction.scene_graph_helpers_psg.dataset.augmentation_utils import get_crop_image_augmentations
from scene_graph_prediction.scene_graph_helpers_psg.dataset.structures import nested_tensor_from_tensor_list
from scene_graph_prediction.scene_graph_helpers_psg.model.model_utils import get_image_model
from scene_graph_prediction.scene_graph_helpers_psg.model.pix2sg_model.detr_configs import N_ENTITIES, N_PREDS, ENTITY_START, INSTANCE_START


class PSGDataset(Dataset):
    def __init__(self,
                 config,
                 split='train'):

        self.split = split
        with ANN_FILE_PATH.open() as f:
            self.ann_json = json.load(f)

        for d in self.ann_json['data']:
            for r in d['relations']:
                r[2] += 1

        # NOTE: Filter out images with zero relations.
        # Comment out this part for competition files
        self.ann_json['data'] = [d for d in self.ann_json['data'] if len(d['relations']) != 0]

        # Get split
        assert split in {'train', 'test'}
        if split == 'train':
            self.data = [d for d in self.ann_json['data']
                         if d['image_id'] not in self.ann_json['test_image_ids']
                         ]
            # self.data = self.data[:1000] # for quick debug
        elif split == 'test':
            self.data = [
                d for d in self.ann_json['data']
                if d['image_id'] in self.ann_json['test_image_ids']
            ]
            # self.data = self.data[:1000] # for quick debug
        # Init image infos
        self.data_infos = []
        for d in self.data:
            self.data_infos.append({
                'filename': d['file_name'],
                'height': d['height'],
                'width': d['width'],
                'id': d['image_id'],
            })
        self.img_ids = [d['id'] for d in self.data_infos]

        # Define classes, 0-index
        # NOTE: Class ids should range from 0 to (num_classes - 1)
        self.THING_CLASSES = self.ann_json['thing_classes']
        self.STUFF_CLASSES = self.ann_json['stuff_classes']
        self.CLASSES = self.THING_CLASSES + self.STUFF_CLASSES
        self.PREDICATES = ['__background__'] + self.ann_json['predicate_classes']

        self.ind_to_classes = {i: c for i, c in enumerate(self.CLASSES)}
        self.ind_to_predicates = {i: p for i, p in enumerate(self.PREDICATES)}
        # # calculate max number of instances
        # max_num_instances = []
        # for elem in self.data:
        #     used_annotation_indices = set()
        #     for relation in elem['relations']:
        #         used_annotation_indices.add(relation[0])
        #         used_annotation_indices.add(relation[1])
        #     list_of_category_ids = [elem['annotations'][i]['category_id'] for i in used_annotation_indices]
        #     highest_num_instances = Counter(list_of_category_ids).most_common(1)[0][1]
        #     max_num_instances.append(highest_num_instances)
        #
        #     # max_num_instances = max(max_num_instances, highest_num_instances)

        assert len(self.ind_to_classes) == N_ENTITIES
        assert len(self.ind_to_predicates) == N_PREDS

        self.image_transformations = get_image_model(model_config=config['MODEL'], only_transforms=True)
        if self.image_transformations is not None:
            self.image_transformations = self.image_transformations[split]
            self.image_augmentations = get_crop_image_augmentations(config, split, for_eval=False)

    def collate_fn(self, batch):
        images_list = [elem['full_image'] for elem in batch]
        image_tensors = nested_tensor_from_tensor_list(images_list)
        targets = []
        for elem in batch:
            elem['target']['relation'] = torch.tensor(elem['relation'])
            elem['target']['index'] = elem['index']
            elem['target']['image_name'] = elem['image_name']
            targets.append(elem['target'])
        return tuple([image_tensors, targets])

    def get_ann_info(self, idx):
        d = self.data[idx]

        # Process bbox annotations
        gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)
        gt_bboxes = np.array([a['bbox'] for a in d['annotations']],
                             dtype=np.float32)
        gt_labels = np.array([a['category_id'] for a in d['annotations']],
                             dtype=np.int64)
        # Process segment annotations
        gt_mask_infos = []
        for s in d['segments_info']:
            gt_mask_infos.append({
                'id': s['id'],
                'category': s['category_id'],
                'is_thing': s['isthing']
            })

        # Process relationship annotations
        gt_rels = d['relations'].copy()

        # Filter out dupes!
        if self.split == 'train':
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in gt_rels:
                all_rel_sets[(o0, o1)].append(r)
            gt_rels = [(k[0], k[1], np.random.choice(v))
                       for k, v in all_rel_sets.items()]
            gt_rels = np.array(gt_rels, dtype=np.int32)
        else:
            # for test or val set, filter the duplicate triplets,
            # but allow multiple labels for each pair
            all_rel_sets = []
            for (o0, o1, r) in gt_rels:
                if (o0, o1, r) not in all_rel_sets:
                    all_rel_sets.append((o0, o1, r))
            gt_rels = np.array(all_rel_sets, dtype=np.int32)

        # add relation to target
        num_box = len(gt_mask_infos)
        relation_map = np.zeros((num_box, num_box), dtype=np.int64)
        for i in range(gt_rels.shape[0]):
            # If already exists a relation?
            if relation_map[int(gt_rels[i, 0]), int(gt_rels[i, 1])] > 0:
                if random.random() > 0.5:
                    relation_map[int(gt_rels[i, 0]),
                    int(gt_rels[i, 1])] = int(gt_rels[i, 2])
            else:
                relation_map[int(gt_rels[i, 0]),
                int(gt_rels[i, 1])] = int(gt_rels[i, 2])

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            rels=gt_rels,
            rel_maps=relation_map,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_mask_infos,
            seg_map=d['pan_seg_file_name'],
        )

        return ann

    def get_entity_instance_id(self, entity, entity_idx, entity_and_idx_to_instance_id):
        '''
        Get the instance id of the entity.
        Importantly: This function will update the entity_and_idx_to_instance_id dict.
        '''
        if entity not in entity_and_idx_to_instance_id:
            instance_id = INSTANCE_START
            entity_and_idx_to_instance_id[entity] = {entity_idx: instance_id}
        else:
            if entity_idx not in entity_and_idx_to_instance_id[entity]:
                instance_id = max(entity_and_idx_to_instance_id[entity].values()) + 1
                entity_and_idx_to_instance_id[entity][entity_idx] = instance_id
            else:
                instance_id = entity_and_idx_to_instance_id[entity][entity_idx]

        return instance_id

    def __getitem__(self, index):
        ann = self.get_ann_info(index)
        img = Image.open(DATA_ROOT_PATH / 'coco' / self.data[index]['file_name']).convert("RGB")
        w, h = img.size
        if img.size[0] != self.data[index]['width'] or img.size[1] != self.data[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.data[index]['width']), ' ', str(self.data[index]['height']),
                  ' ', '=' * 20)
        # Classes go to 151, Predicates go to 50

        target = {}
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        # self.plot_bounding_boxes(img, target['boxes'], target['entity_labels'])
        relations = []
        for sub_idx, obj_idx, rel_idx in ann['rels']:
            sub = ann['labels'][sub_idx]
            obj = ann['labels'][obj_idx]
            relations.append((sub + ENTITY_START, sub_idx, rel_idx, obj + ENTITY_START, obj_idx))

        random.shuffle(relations)

        if self.image_augmentations is not None:
            img = self.image_augmentations(img)
        if isinstance(self.image_transformations, DetrFeatureExtractor):
            img = self.image_transformations(images=img, return_tensors="pt")['pixel_values'][0]
        else:
            img = self.image_transformations(img)

        # remove boxes and entity_labels from target. Instead just use the relation list
        entity_to_instance_to_bbox = {}
        relation_instances = []
        for (sub_id, sub_index, rel_id, obj_id, obj_index) in relations:
            sub_instance = self.get_entity_instance_id(sub_id, sub_index, entity_to_instance_to_bbox)
            obj_instance = self.get_entity_instance_id(obj_id, obj_index, entity_to_instance_to_bbox)
            rel_tpl = (sub_id, sub_instance, rel_id, obj_id, obj_instance)
            relation_instances.append(rel_tpl)

        relations = relation_instances

        return {'full_image': img, 'target': target, 'relation': relations, 'index': index, 'image_name': self.data[index]['file_name']}

    def __len__(self):
        return len(self.data)
