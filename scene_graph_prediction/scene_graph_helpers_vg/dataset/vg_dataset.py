import json
import os
import random
from collections import defaultdict

import h5py as h5py
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt, patches
from torch.utils.data import Dataset
from torchvision.ops import box_iou
# import torch.utils.data as data
# from torchvision import transforms as T
from tqdm import tqdm

from helpers.vg_helpers.configurations import IMG_FILE_PATH, IMG_DIR_PATH, DICT_FILE_PATH, ROIDB_FILE_PATH
from scene_graph_prediction.scene_graph_helpers_vg.dataset.augmentation_utils import get_crop_image_augmentations
from scene_graph_prediction.scene_graph_helpers_vg.dataset.structures import BoxList, boxlist_iou, nested_tensor_from_tensor_list
from scene_graph_prediction.scene_graph_helpers_vg.model.model_utils import get_image_model
from scene_graph_prediction.scene_graph_helpers_vg.model.pix2sg_model.detr_configs import N_ENTITIES, N_PREDS, ENTITY_START, INSTANCE_START

BOX_SCALE = 1024


class VGDataset(Dataset):
    def __init__(self,
                 config,
                 split='train',
                 for_eval=False,
                 filter_duplicate_rels=True, num_im=-1, num_val_im=5000, filter_empty_rels=True):

        assert split in {'train', 'val', 'test'}
        self.split = split
        self.img_dir = IMG_DIR_PATH
        self.dict_file = DICT_FILE_PATH
        self.roidb_file = ROIDB_FILE_PATH
        self.image_file = IMG_FILE_PATH
        self.filter_non_overlap = False
        self.filter_duplicate_rels = filter_duplicate_rels and self.split == 'train'
        # self.resize_transform = T.Resize(max_size=)
        self.image_transformations = get_image_model(model_config=config['MODEL'], only_transforms=True)
        if self.image_transformations is not None:
            self.image_transformations = self.image_transformations[split]
            self.image_augmentations = get_crop_image_augmentations(config, self.split, for_eval=False)

        # self.transforms = self.make_transforms(config, split)
        self.ind_to_classes, self.ind_to_predicates, self.ind_to_attributes = load_info(self.dict_file)  # contiguous 151, 51 containing __background__
        assert len(self.ind_to_classes) == N_ENTITIES
        assert len(self.ind_to_predicates) == N_PREDS
        self.categories = {i: self.ind_to_classes[i] for i in range(len(self.ind_to_classes))}

        self.split_mask, self.gt_boxes, self.gt_classes, self.gt_attributes, self.relationships = load_graphs(
            self.roidb_file, self.split, num_im, num_val_im=num_val_im,
            filter_empty_rels=filter_empty_rels,
            filter_non_overlap=self.filter_non_overlap,
        )

        self.filenames, self.img_info = load_image_filenames(self.img_dir, self.image_file)  # length equals to split_mask
        self.filenames = [self.filenames[i] for i in np.where(self.split_mask)[0]]
        self.img_info = [self.img_info[i] for i in np.where(self.split_mask)[0]]

    def collate_fn(self, batch):
        images_list = [elem['full_image'] for elem in batch]
        image_tensors = nested_tensor_from_tensor_list(images_list)
        targets = []
        for elem in batch:
            elem['target']['relation'] = torch.tensor(elem['relation'])
            elem['target']['index'] = elem['index']
            elem['target']['image_id'] = elem['image_id']
            targets.append(elem['target'])
        return tuple([image_tensors, targets])

    def __getitem__(self, index):
        img = Image.open(self.filenames[index]).convert("RGB")
        w, h = img.size
        if img.size[0] != self.img_info[index]['width'] or img.size[1] != self.img_info[index]['height']:
            print('=' * 20, ' ERROR index ', str(index), ' ', str(img.size), ' ', str(self.img_info[index]['width']), ' ', str(self.img_info[index]['height']),
                  ' ', '=' * 20)
        # Classes go to 151, Predicates go to 50
        _target = self.get_groundtruth(index)
        target = {}
        target['boxes'] = _target.bbox
        target['entity_labels'] = _target.extra_fields['labels']  # entity_labels
        target['size'] = torch.as_tensor([int(h), int(w)])
        target['orig_size'] = torch.as_tensor([int(h), int(w)])
        # self.plot_bounding_boxes(img, target['boxes'], target['entity_labels'])
        relation = self.relationships[index]  # here format is: [sub_index,obj_index, rel_index]
        # Add + len(self.ind_to_predicates) to avoid conflicts between pred tokens and entity tokens
        relation = [(int(target['entity_labels'][sub_index]) + ENTITY_START, sub_index, rel_index,
                     int(target['entity_labels'][obj_index]) + ENTITY_START, obj_index) for (sub_index, obj_index, rel_index) in
                    relation]  # here format is: [sub_id,rel_id,obj_id]
        random.shuffle(relation)

        if self.image_augmentations is not None:
            img = self.image_augmentations(img)
        img = self.image_transformations(img)

        # remove boxes and entity_labels from target. Instead just use the relation list
        entity_to_instance_to_bbox = {}
        relation_unique_instances = []
        for (sub_id, sub_index, rel_id, obj_id, obj_index) in relation:
            # As other things are below, use the range from 250
            if sub_id not in entity_to_instance_to_bbox:
                sub_instance = INSTANCE_START
                entity_to_instance_to_bbox[sub_id] = {sub_instance: target['boxes'][sub_index]}
            else:
                for instance, bbox in entity_to_instance_to_bbox[sub_id].items():
                    # if torch.equal(bbox, target['boxes'][sub_index]):
                    if self.is_bounding_box_equal(bbox, target['boxes'][sub_index]):
                        sub_instance = instance
                        break
                else:
                    sub_instance = max(entity_to_instance_to_bbox[sub_id].keys()) + 1
                    entity_to_instance_to_bbox[sub_id][sub_instance] = target['boxes'][sub_index]

            if obj_id not in entity_to_instance_to_bbox:
                obj_instance = INSTANCE_START
                entity_to_instance_to_bbox[obj_id] = {obj_instance: target['boxes'][obj_index]}
            else:
                for instance, bbox in entity_to_instance_to_bbox[obj_id].items():
                    # if torch.equal(bbox, target['boxes'][obj_index]):
                    if self.is_bounding_box_equal(bbox, target['boxes'][obj_index]):
                        obj_instance = instance
                        break
                else:
                    obj_instance = max(entity_to_instance_to_bbox[obj_id].keys()) + 1
                    entity_to_instance_to_bbox[obj_id][obj_instance] = target['boxes'][obj_index]

            rel_tpl = (sub_id, sub_instance, rel_id, obj_id, obj_instance)
            relation_unique_instances.append(rel_tpl)
            # if rel_tpl not in relation_unique_instances: # optionally remove duplicates like this
            #     relation_unique_instances.append(rel_tpl)

        relation = relation_unique_instances

        del target['boxes']
        del target['entity_labels']

        return {'full_image': img, 'target': target, 'relation': relation, 'index': index, 'image_id': self.img_info[index]['image_id']}

    def get_statistics(self):
        fg_matrix, bg_matrix = get_VG_statistics(img_dir=self.img_dir, roidb_file=self.roidb_file, dict_file=self.dict_file,
                                                 image_file=self.image_file, must_overlap=True)
        eps = 1e-3
        bg_matrix += 1
        fg_matrix[:, :, 0] = bg_matrix
        pred_dist = np.log(fg_matrix / fg_matrix.sum(2)[:, :, None] + eps)

        result = {
            'fg_matrix': torch.from_numpy(fg_matrix),
            'pred_dist': torch.from_numpy(pred_dist).float(),
            'obj_classes': self.ind_to_classes,
            'rel_classes': self.ind_to_predicates,
            'att_classes': self.ind_to_attributes,
        }
        return result

    def is_bounding_box_equal(self, box1: torch.FloatTensor, box2: torch.FloatTensor, IoU_threshold=0.5):
        Iou = box_iou(box1.unsqueeze(0), box2.unsqueeze(0))[0]
        return Iou > IoU_threshold

    def plot_bounding_boxes(self, img, boxes, labels):
        # high res plot
        fig, ax = plt.subplots(1, figsize=(10, 10 * img.size[1] / img.size[0]))
        ax.imshow(img)
        for box, label in zip(boxes, labels):
            rect = patches.Rectangle((box[0], box[1]), box[2] - box[0], box[3] - box[1], linewidth=2, edgecolor='r',
                                     facecolor='none')
            ax.add_patch(rect)
            ax.text(box[0], box[1], self.ind_to_classes[label], fontsize=24, color='r')
        plt.savefig('debug_img.jpg')

    def get_custom_imgs(self, path):
        self.custom_files = []
        self.img_info = []
        if os.path.isdir(path):
            for file_name in tqdm(os.listdir(path)):
                self.custom_files.append(os.path.join(path, file_name))
                img = Image.open(os.path.join(path, file_name)).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height)})
        # Expecting a list of paths in a json file
        if os.path.isfile(path):
            file_list = json.load(open(path))
            for file in tqdm(file_list):
                self.custom_files.append(file)
                img = Image.open(file).convert("RGB")
                self.img_info.append({'width': int(img.width), 'height': int(img.height)})

    def get_img_info(self, index):
        # WARNING: original image_file.json has several pictures with false image size
        # use correct function to check the validity before training
        # it will take a while, you only need to do it once

        # correct_img_info(self.img_dir, self.image_file)
        return self.img_info[index]

    def get_groundtruth(self, index, evaluation=False):
        img_info = self.get_img_info(index)
        w, h = img_info['width'], img_info['height']
        # important: recover original box from BOX_SCALE
        box = self.gt_boxes[index] / BOX_SCALE * max(w, h)
        box = torch.from_numpy(box).reshape(-1, 4)  # guard against no boxes
        target = BoxList(box, (w, h), 'xyxy')  # xyxy

        target.add_field("labels", torch.from_numpy(self.gt_classes[index]))
        target.add_field("attributes", torch.from_numpy(self.gt_attributes[index]))

        relation = self.relationships[index].copy()  # (num_rel, 3)
        if self.filter_duplicate_rels:
            # Filter out dupes!
            assert self.split == 'train'
            old_size = relation.shape[0]
            all_rel_sets = defaultdict(list)
            for (o0, o1, r) in relation:
                all_rel_sets[(o0, o1)].append(r)
            relation = [(k[0], k[1], np.random.choice(v)) for k, v in all_rel_sets.items()]
            relation = np.array(relation, dtype=np.int32)

        # add relation to target
        num_box = len(target)
        relation_map = torch.zeros((num_box, num_box), dtype=torch.int64)
        for i in range(relation.shape[0]):
            if relation_map[int(relation[i, 0]), int(relation[i, 1])] > 0:
                if (random.random() > 0.5):
                    relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
            else:
                relation_map[int(relation[i, 0]), int(relation[i, 1])] = int(relation[i, 2])
        target.add_field("relation", relation_map, is_triplet=True)

        if evaluation:
            target = target.clip_to_image(remove_empty=False)
            target.add_field("relation_tuple", torch.LongTensor(relation))  # for evaluation
            return target
        else:
            # target = target.clip_to_image(remove_empty=True)
            target = target.clip_to_image(remove_empty=False)
            return target

    def __len__(self):
        return len(self.filenames)


def box_filter(boxes, must_overlap=False):
    """ Only include boxes that overlap as possible relations.
    If no overlapping boxes, use all of them."""
    n_cands = boxes.shape[0]

    overlaps = bbox_overlaps(boxes.astype(np.float), boxes.astype(np.float), to_move=0) > 0
    np.fill_diagonal(overlaps, 0)

    all_possib = np.ones_like(overlaps, dtype=np.bool)
    np.fill_diagonal(all_possib, 0)

    if must_overlap:
        possible_boxes = np.column_stack(np.where(overlaps))

        if possible_boxes.size == 0:
            possible_boxes = np.column_stack(np.where(all_possib))
    else:
        possible_boxes = np.column_stack(np.where(all_possib))
    return possible_boxes


def bbox_overlaps(boxes1, boxes2, to_move=1):
    """
    boxes1 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    boxes2 : numpy, [num_obj, 4] (x1,y1,x2,y2)
    """
    # print('boxes1: ', boxes1.shape)
    # print('boxes2: ', boxes2.shape)
    num_box1 = boxes1.shape[0]
    num_box2 = boxes2.shape[0]
    lt = np.maximum(boxes1.reshape([num_box1, 1, -1])[:, :, :2], boxes2.reshape([1, num_box2, -1])[:, :, :2])  # [N,M,2]
    rb = np.minimum(boxes1.reshape([num_box1, 1, -1])[:, :, 2:], boxes2.reshape([1, num_box2, -1])[:, :, 2:])  # [N,M,2]

    wh = (rb - lt + to_move).clip(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    return inter


def correct_img_info(img_dir, image_file):
    with open(image_file, 'r') as f:
        data = json.load(f)
    for i in range(len(data)):
        img = data[i]
        basename = '{}.jpg'.format(img['image_id'])
        filename = os.path.join(img_dir, basename)
        img_data = Image.open(filename).convert("RGB")
        if img['width'] != img_data.size[0] or img['height'] != img_data.size[1]:
            print('--------- False id: ', i, '---------')
            print(img_data.size)
            print(img)
            data[i]['width'] = img_data.size[0]
            data[i]['height'] = img_data.size[1]
    with open(image_file, 'w') as outfile:
        json.dump(data, outfile)


def load_info(dict_file, add_bg=True):
    """
    Loads the file containing the visual genome label meanings
    """
    info = json.load(open(dict_file, 'r'))
    if add_bg:
        info['label_to_idx']['__background__'] = 0
        info['predicate_to_idx']['__background__'] = 0
        info['attribute_to_idx']['__background__'] = 0

    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']
    attribute_to_ind = info['attribute_to_idx']
    ind_to_classes = sorted(class_to_ind, key=lambda k: class_to_ind[k])
    ind_to_predicates = sorted(predicate_to_ind, key=lambda k: predicate_to_ind[k])
    ind_to_attributes = sorted(attribute_to_ind, key=lambda k: attribute_to_ind[k])

    return ind_to_classes, ind_to_predicates, ind_to_attributes


def load_image_filenames(img_dir, image_file):
    """
    Loads the image filenames from visual genome from the JSON file that contains them.
    This matches the preprocessing in scene-graph-TF-release/data_tools/vg_to_imdb.py.
    Parameters:
        image_file: JSON file. Elements contain the param "image_id".
        img_dir: directory where the VisualGenome images are located
    Return:
        List of filenames corresponding to the good images
    """
    with open(image_file, 'r') as f:
        im_data = json.load(f)

    corrupted_ims = ['1592.jpg', '1722.jpg', '4616.jpg', '4617.jpg']
    fns = []
    img_info = []
    for i, img in enumerate(im_data):
        basename = '{}.jpg'.format(img['image_id'])
        if basename in corrupted_ims:
            continue

        filename = os.path.join(img_dir, basename)
        if os.path.exists(filename):
            fns.append(filename)
            img_info.append(img)
    assert len(fns) == 108073
    assert len(img_info) == 108073
    return fns, img_info


def load_graphs(roidb_file, split, num_im, num_val_im, filter_empty_rels, filter_non_overlap):
    # and seperationg the instances, as these could have been encoded in the scene graph as well
    """
    Load the file containing the GT boxes and relations, as well as the dataset split
    Parameters:
        roidb_file: HDF5
        split: (train, val, or test)
        num_im: Number of images we want
        num_val_im: Number of validation images
        filter_empty_rels: (will be filtered otherwise.)
        filter_non_overlap: If training, filter images that dont overlap.
    Return:
        image_index: numpy array corresponding to the index of images we're using
        boxes: List where each element is a [num_gt, 4] array of ground
                    truth boxes (x1, y1, x2, y2)
        gt_classes: List where each element is a [num_gt] array of classes
        relationships: List where each element is a [num_r, 3] array of
                    (box_ind_1, box_ind_2, predicate) relationships
    """
    roi_h5 = h5py.File(roidb_file, 'r')
    data_split = roi_h5['split'][:]
    split_flag = 2 if split == 'test' else 0
    split_mask = data_split == split_flag

    # Filter out images without bounding boxes
    split_mask &= roi_h5['img_to_first_box'][:] >= 0
    if filter_empty_rels:
        split_mask &= roi_h5['img_to_first_rel'][:] >= 0

    image_index = np.where(split_mask)[0]
    if num_im > -1:
        image_index = image_index[:num_im]
    if num_val_im > 0:
        if split == 'val':
            image_index = image_index[:num_val_im]
        elif split == 'train':
            image_index = image_index[num_val_im:]

    split_mask = np.zeros_like(data_split).astype(bool)
    split_mask[image_index] = True

    # Get box information
    all_labels = roi_h5['labels'][:, 0]
    all_attributes = roi_h5['attributes'][:, :]
    all_boxes = roi_h5['boxes_{}'.format(BOX_SCALE)][:]  # cx,cy,w,h
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box

    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box'][split_mask]
    im_to_last_box = roi_h5['img_to_last_box'][split_mask]
    im_to_first_rel = roi_h5['img_to_first_rel'][split_mask]
    im_to_last_rel = roi_h5['img_to_last_rel'][split_mask]

    # load relation labels
    _relations = roi_h5['relationships'][:]
    _relation_predicates = roi_h5['predicates'][:, 0]
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    assert (_relations.shape[0] == _relation_predicates.shape[0])  # sanity check

    # Get everything by image.
    boxes = []
    gt_classes = []
    gt_attributes = []
    relationships = []
    for i in range(len(image_index)):
        i_obj_start = im_to_first_box[i]
        i_obj_end = im_to_last_box[i]
        i_rel_start = im_to_first_rel[i]
        i_rel_end = im_to_last_rel[i]

        boxes_i = all_boxes[i_obj_start: i_obj_end + 1, :]
        gt_classes_i = all_labels[i_obj_start: i_obj_end + 1]
        gt_attributes_i = all_attributes[i_obj_start: i_obj_end + 1, :]

        if i_rel_start >= 0:
            predicates = _relation_predicates[i_rel_start: i_rel_end + 1]
            obj_idx = _relations[i_rel_start: i_rel_end + 1] - i_obj_start  # range is [0, num_box)
            assert np.all(obj_idx >= 0)
            assert np.all(obj_idx < boxes_i.shape[0])
            rels = np.column_stack((obj_idx, predicates))  # (num_rel, 3), representing sub, obj, and pred
        else:
            assert not filter_empty_rels
            rels = np.zeros((0, 3), dtype=np.int32)

        if filter_non_overlap:
            assert split == 'train'
            # construct BoxList object to apply boxlist_iou method
            # give a useless (height=0, width=0)
            boxes_i_obj = BoxList(boxes_i, (1000, 1000), 'xyxy')
            inters = boxlist_iou(boxes_i_obj, boxes_i_obj)
            rel_overs = inters[rels[:, 0], rels[:, 1]]
            inc = np.where(rel_overs > 0.0)[0]

            if inc.size > 0:
                rels = rels[inc]
            else:
                split_mask[image_index[i]] = 0
                continue

        boxes.append(boxes_i)
        gt_classes.append(gt_classes_i)
        gt_attributes.append(gt_attributes_i)
        relationships.append(rels)

    return split_mask, boxes, gt_classes, gt_attributes, relationships
