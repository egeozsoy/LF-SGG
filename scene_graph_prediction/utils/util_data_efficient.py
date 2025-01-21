import numpy as np
import open3d as o3d
import random
import torch


def build_edge_from_selection(node_ids, max_edges_per_node):
    '''
    flow: an edge passes message from i to j is denoted as  [i,j]. 
    '''
    ''' build trees '''
    edge_indices = list()
    for s_idx in node_ids:
        nn = [elem for elem in node_ids if elem != s_idx]  # every node except itself
        if max_edges_per_node > 0:
            if len(nn) > max_edges_per_node:
                nn = list(np.random.choice(list(nn), max_edges_per_node))

        for t_idx in nn:
            edge_indices.append([s_idx, t_idx])
    return edge_indices


def zero_mean(point):
    mean = torch.mean(point, dim=0)
    point -= mean.unsqueeze(0)
    # Norm to unit sphere
    furthest_distance = point.pow(2).sum(1).sqrt().max()  # find maximum distance for each n -> [n]
    point /= furthest_distance
    return point, {'mean': mean.unsqueeze(0), 'dist': furthest_distance}


def objname_to_onehot(objname):
    obj_name_to_index = {
        'anesthesia_equipment': 0,
        'operating_table': 1,
        'instrument_table': 2,
        'secondary_table': 3,
        'instrument': 4,
        # 'object': 5,
        'human': 5
    }
    if 'human' in objname or 'Patient' in objname:  # We don't care about patient human_0 human_1 etc. everything is human (We don't seperate patient here, because voxelpose also won't seperate it)
        objname = 'human'
    one_hot_vector = np.zeros(len(obj_name_to_index))
    one_hot_vector[obj_name_to_index[objname]] = 1
    return one_hot_vector


def calculate_downsample_indices(pointset, target_N):
    if len(pointset) < target_N:
        return np.random.choice(len(pointset), target_N, replace=True)

    pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pointset[:, :3]))
    best_choice = np.asarray(list(range(len(pointset))))
    for sampling_size in range(15, 100, 5):
        choice = np.unique(pc.voxel_down_sample_and_trace(sampling_size, pc.get_min_bound(), pc.get_max_bound())[1])[1:]
        if len(choice) > target_N:
            best_choice = choice
        else:
            break
    return best_choice[np.random.choice(len(best_choice), target_N, replace=False)]


def convert_multi_rel_to_name(rel, relation_names):
    all_rels = np.asarray(relation_names)[rel == 1].tolist()
    if len(all_rels) == 0:
        return 'none'
    elif len(all_rels) == 1:
        return all_rels[0]
    else:
        all_rels.remove('CloseTo')
        if len(all_rels) == 1:
            return all_rels[0]
        else:
            all_rels.remove('Touching')
            return all_rels[0]


def convert_rel_multi_to_single(rel, relation_names):
    rel = rel.detach()
    np_relation_names = np.asarray(relation_names)
    mask = (rel > 0.5).cpu()
    single_rel = torch.zeros(rel.shape[0], rel.shape[1] + 1)
    for idx, (r, m) in enumerate(zip(rel, mask)):
        valid = np_relation_names[:-1][m].tolist()
        if len(valid) == 0:
            rel_name = 'none'
        elif len(valid) == 1:
            rel_name = valid[0]
        else:
            if 'CloseTo' in valid: valid.remove('CloseTo')
            if len(valid) == 1:
                rel_name = valid[0]
            else:
                if 'Touching' in valid: valid.remove('Touching')
                if len(valid) == 1:
                    rel_name = valid[0]
                else:
                    # Pick the biggest score
                    max_score = -10000
                    max_v = None
                    for v in valid:
                        score = r[relation_names.index(v)]
                        if score > max_score:
                            max_score = score
                            max_v = v
                    rel_name = max_v
        single_rel[idx][relation_names.index(rel_name)] = 1
    return single_rel


def data_preparation(points, instances, selected_instances, num_points,
                     # use_rgb, use_normal,
                     for_train=False, instance2labelName=None, classNames=None,
                     rel_json=None, relationships=None, multi_rel_outputs=None,
                     padding=0.2, num_max_rel=-1, shuffle_objs=True,
                     sample_in_runtime: bool = False, num_nn=1, num_seed=1, instance_label_to_hand_locations=None):
    if for_train:
        assert (instance2labelName is not None)
        assert (rel_json is not None)
        assert (classNames is not None)
        assert (relationships is not None)
        assert (multi_rel_outputs is not None)

    instances = instances.flatten()

    instances_id = list(np.unique(instances))

    if 0 in instances_id:
        instances_id.remove(0)
    if shuffle_objs:
        random.shuffle(instances_id)

    instance2mask = {}
    instance2mask[0] = 0
    cat = []
    counter = 0
    ''' Build instance2mask and their gt classes '''
    for instance_id in list(np.unique(instances)):
        # print('instance {} size: {}'.format(instance_id,len(points[np.where(instances == instance_id)])))
        if selected_instances is not None:
            if instance_id not in selected_instances:
                # since we divide the whole graph into sub-graphs if the 
                # scene graph is too large to resolve the memory issue, there 
                # are instances that are not interested in the current sub-graph
                instance2mask[instance_id] = 0
                continue

        if for_train:
            class_id = -1  # was 0

            instance_labelName = instance2labelName[instance_id]
            if instance_labelName in classNames:  # is it a class we care about?
                class_id = classNames.index(instance_labelName)

            if (class_id >= 0) and (instance_id > 0):  # there is no instance 0?
                cat.append(class_id)
        else:
            class_id = 0

        if class_id != -1:  # was 0
            counter += 1
            instance2mask[instance_id] = counter
        else:
            instance2mask[instance_id] = 0

    num_objects = len(instances_id) if selected_instances is None else len(selected_instances)
    masks = np.asarray([instance2mask[instance] for instance in instances], dtype=np.int32)
    mask2_instance = {v: k for k, v in instance2mask.items()}
    dim_point = points.shape[1]

    # create normalized pointsets for each object, sorted like the masks
    obj_bboxes = list()
    obj_masks = list()
    for i in range(num_objects):
        points_mask = torch.zeros(points.shape[0], dtype=torch.bool)
        points_mask[np.where(masks == i + 1)[0]] = True
        obj_pointset = points[np.where(masks == i + 1)[0], :]
        min_box = np.min(obj_pointset[:, :3], 0) - padding
        max_box = np.max(obj_pointset[:, :3], 0) + padding
        obj_bboxes.append([min_box, max_box])
        obj_masks.append(points_mask)
    # Build fully connected edges
    edge_indices = list()
    max_edges = -1
    for n in range(len(cat)):
        for m in range(len(cat)):
            if n == m: continue
            edge_indices.append([n, m])
    if max_edges > 0 and len(edge_indices) > max_edges and for_train:
        # for eval, do not drop out any edges.
        indices = list(np.random.choice(len(edge_indices), max_edges))
        edge_indices = edge_indices[indices]

    if for_train:
        ''' Build rel class GT '''
        if multi_rel_outputs:
            adj_matrix_onehot = np.zeros([num_objects, num_objects, len(relationships)])
        else:
            adj_matrix = np.zeros([num_objects, num_objects]) + relationships.index('none')  # Default should be none

        for r in rel_json:
            if r[0] not in instance2mask or r[1] not in instance2mask: continue
            index1 = instance2mask[r[0]] - 1
            index2 = instance2mask[r[1]] - 1
            if for_train:
                if r[3] not in relationships:
                    continue
                r[2] = relationships.index(r[3])  # remap the index of relationships in case of custom relationNames

            if index1 >= 0 and index2 >= 0:
                if multi_rel_outputs:
                    adj_matrix_onehot[index1, index2, r[2]] = 1
                else:
                    adj_matrix[index1, index2] = r[2]

        ''' Build rel point cloud '''
        if multi_rel_outputs:
            rel_dtype = np.float32
            adj_matrix_onehot = torch.from_numpy(np.array(adj_matrix_onehot, dtype=rel_dtype))
        else:
            rel_dtype = np.int64

        if multi_rel_outputs:
            gt_rels = torch.zeros(len(edge_indices), len(relationships), dtype=torch.float)
        else:
            gt_rels = torch.zeros(len(edge_indices), dtype=torch.long)

    rel_masks = list()
    rel_hand_points = torch.zeros(len(edge_indices), 2, 3, dtype=torch.float16)
    relation_objects_one_hot = []
    for e in range(len(edge_indices)):
        edge = edge_indices[e]
        index1 = edge[0]
        index2 = edge[1]
        obj1_instance = mask2_instance[index1 + 1]
        obj2_instance = mask2_instance[index2 + 1]
        obj1_name = instance2labelName[obj1_instance]
        obj2_name = instance2labelName[obj2_instance]
        if for_train:
            if multi_rel_outputs:
                gt_rels[e, :] = adj_matrix_onehot[index1, index2, :]
            else:
                gt_rels[e] = adj_matrix[index1, index2]
                # rel_name = relationships[gt_rels[e].item()]
                # if rel_name != 'none':
                #     print(f'{obj1_name} -{rel_name}> {obj2_name}')

        obj1_one_hot = objname_to_onehot(obj1_name)
        obj2_one_hot = objname_to_onehot(obj2_name)
        relation_objects_one_hot.append(np.concatenate([obj1_one_hot, obj2_one_hot]))

        bbox1 = obj_bboxes[index1]
        bbox2 = obj_bboxes[index2]
        min_box = np.minimum(bbox1[0], bbox2[0])
        max_box = np.maximum(bbox1[1], bbox2[1])
        filter_mask = (points[:, 0] > min_box[0]) * (points[:, 0] < max_box[0]) \
                      * (points[:, 1] > min_box[1]) * (points[:, 1] < max_box[1]) \
                      * (points[:, 2] > min_box[2]) * (points[:, 2] < max_box[2])
        points_mask = torch.zeros(points.shape[0], dtype=torch.bool)
        points_mask[filter_mask] = True
        rel_masks.append(points_mask)

    relation_objects_one_hot = torch.from_numpy(np.stack(relation_objects_one_hot)).float()
    cat = torch.from_numpy(np.array(cat, dtype=np.int64))
    edge_indices = torch.tensor(edge_indices, dtype=torch.long)

    obj_masks = torch.stack(obj_masks)
    rel_masks = torch.stack(rel_masks)

    mask_floor = points[:, 1] > 100
    points = points[mask_floor]  # remove the floor
    obj_masks = obj_masks[:, mask_floor]  # remove the floor
    rel_masks = rel_masks[:, mask_floor]  # remove the floor
    mask_side1 = points[:, 2] > -950
    points = points[mask_side1]  # remove unnecessary side
    obj_masks = obj_masks[:, mask_side1]  # remove unnecessary side
    rel_masks = rel_masks[:, mask_side1]  # remove unnecessary side
    mask_side2 = points[:, 0] > -600
    points = points[mask_side2]  # remove unnecessary side
    obj_masks = obj_masks[:, mask_side2]  # remove unnecessary side
    rel_masks = rel_masks[:, mask_side2]  # remove unnecessary side
    choice = calculate_downsample_indices(points, num_points)  # TODO maybe do uniform sampling
    points = points[choice, :]
    obj_masks = obj_masks[:, choice]
    rel_masks = rel_masks[:, choice]
    points = torch.from_numpy(points.astype(np.float32))

    points[:, :3] /= 1000.0

    if for_train:
        return points, obj_masks, rel_masks, edge_indices, instance2mask, relation_objects_one_hot, gt_rels, cat, rel_hand_points
    else:
        return points, obj_masks, rel_masks, edge_indices, instance2mask, relation_objects_one_hot, rel_hand_points
