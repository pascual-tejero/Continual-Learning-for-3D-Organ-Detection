"""Helper functions for handling bounding boxes."""

import torch
import numpy as np
from collections import defaultdict

def generalized_bbox_iou_3d(bboxes1, bboxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, z0, x1, y1. z1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    assert (bboxes1[:, 3:] >= bboxes1[:, :3]).all()
    assert (bboxes2[:, 3:] >= bboxes2[:, :3]).all()

    
    iou, union = iou_3d(bboxes1, bboxes2)

    x1 = torch.min(bboxes1[:, None, 0], bboxes2[:, 0])
    y1 = torch.min(bboxes1[:, None, 1], bboxes2[:, 1])
    z1 = torch.min(bboxes1[:, None, 2], bboxes2[:, 2])
    x2 = torch.max(bboxes1[:, None, 3], bboxes2[:, 3])
    y2 = torch.max(bboxes1[:, None, 4], bboxes2[:, 4])
    z2 = torch.max(bboxes1[:, None, 5], bboxes2[:, 5])

    dx = (x2 - x1).clamp(min=0)
    dy = (y2 - y1).clamp(min=0)
    dz = (z2 - z1).clamp(min=0)

    vol = dx * dy * dz
    return iou - (vol - union) / vol

def box_cxcyczwhd_to_xyzxyz(bboxes):
    if isinstance(bboxes, torch.Tensor):
        x_c, y_c, z_c, w, h, d = bboxes.unbind(-1)
    else:
        x_c, y_c, z_c, w, h, d = bboxes.T

    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (z_c - 0.5 * d),
         (x_c + 0.5 * w), (y_c + 0.5 * h), (z_c + 0.5 * d)]

    if isinstance(bboxes, torch.Tensor):
        return torch.stack(b, dim=-1)
    else:
        return np.stack(b, axis=-1)


def organ_in_boundary(boundaries, class_):
    for boundary in boundaries:
        if class_ in boundary:
            return True

def segmentation2bbox(segmentation_maps, padding, box_format='cxcyczwhd', normalize=True, excl_crossed_boundary=False):
    batch_bboxes = []
    batch_classes = []
    for map_ in segmentation_maps:
        assert map_.ndim == 4

        valid_bboxes = []
        valid_classes = []
        classes = [int(class_) for class_ in map_.unique(sorted=True)][1:]
        for class_ in classes:
            class_indices = (map_ == class_).nonzero(as_tuple=False)

            min_values = class_indices.min(dim=0)[0][1:].to(torch.float)   # x, y, z
            max_values = class_indices.max(dim=0)[0][1:].to(torch.float)

            # Ignore too small boxes
            if ((max_values - min_values) < 5).any():
                continue        
            
            if excl_crossed_boundary:
                margin_boundary = 1
                boundaries = [
                    map_[0, 0:margin_boundary, :, :],
                    map_[0, :, 0:margin_boundary, :],
                    map_[0, :, :, 0:margin_boundary],
                    map_[0, -margin_boundary:, :, :],
                    map_[0, :, -margin_boundary:, :],
                    map_[0, :, :, -margin_boundary:],
                ]
                if organ_in_boundary(boundaries, class_):
                    continue
            
            # Apply padding to bounding boxes
            min_values = (min_values - padding).clip(min=0) 
            max_values = (max_values + padding).clip(max=torch.tensor(map_.shape[1:]))

            assert min_values[0] < max_values[0]
            assert min_values[1] < max_values[1]
            assert min_values[2] < max_values[2]

            if normalize:   # Put coords between 0 and 1; nec for sigmoid output
                min_values /= torch.tensor(map_.shape[1:])
                max_values /= torch.tensor(map_.shape[1:])

            if box_format == 'xyzxyz':
                valid_bboxes.append(torch.hstack((min_values, max_values)))   # x1, y1, z1, x2, y2, z2
            elif box_format == 'xyxyzz':
                valid_bboxes.append(torch.hstack((min_values[0:2], max_values[0:2], min_values[-1], max_values[-1])))
            elif box_format == 'cxcyczwhd':
                width, height, depth = max_values - min_values
                cx, cy, cz = (max_values + min_values) / 2
                valid_bboxes.append(torch.tensor([cx, cy, cz, width, height, depth]))
            else:
                raise ValueError('Please select a valid box format.')

            valid_classes.append(class_)

        batch_classes.append(torch.tensor(valid_classes))
        
        try:
            batch_bboxes.append(torch.vstack(valid_bboxes))
        except:
            batch_bboxes.append(torch.tensor(valid_bboxes))

    return batch_bboxes, batch_classes


def iou_3d(bboxes1, bboxes2):
    """Determines the intersection over union (IoU) for two sets of
    three dimensional bounding boxes.

    Bounding boxes have to be in the format (x1, y1, z1, x2, y2, z2).

    Args:
        bboxes1: A tensor of the shape [N, 6] containing the first
            set of bounding boxes.
        bboxes2: A tensor of the shape [M, 6] containing the first
            set of bounding boxes.

    Returns:
        A tensor of shape [N, M] containing the IoU values of all 
        bounding boxes to each other and a tensor of same shape containing
        the pure union values between bboxes.
    """
    volume_bbox1 = bboxes_volume(bboxes1)
    volume_bbox2 = bboxes_volume(bboxes2)

    x1 = torch.max(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = torch.max(bboxes1[:, None, 1], bboxes2[None, :, 1])
    z1 = torch.max(bboxes1[:, None, 2], bboxes2[None, :, 2])
    x2 = torch.min(bboxes1[:, None, 3], bboxes2[None, :, 3])
    y2 = torch.min(bboxes1[:, None, 4], bboxes2[None, :, 4])
    z2 = torch.min(bboxes1[:, None, 5], bboxes2[None, :, 5])

    delta_x = (x2 - x1).clamp(min=0)
    delta_y = (y2 - y1).clamp(min=0)
    delta_z = (z2 - z1).clamp(min=0)

    intersection = delta_x * delta_y * delta_z
    union = volume_bbox1[:, None] + volume_bbox2 - intersection
    iou = intersection / union

    return iou, union 

def bboxes_volume(bboxes):
    """Estimates the volume of a three dimensional bounding box.
    
    Args:
        bboxes: A tensor of the shape [N, 6] containing N bounding
            boxes in the format (x1, y1, z1, x2, y2, z2).

    Returns:
        A tensor of shape (N,) containing the corresponding volumes.
    """
    delta_x = bboxes[:, 3] - bboxes[:, 0]
    delta_y = bboxes[:, 4] - bboxes[:, 1]
    delta_z = bboxes[:, 5] - bboxes[:, 2]
    return delta_x * delta_y * delta_z

def iou_3d_np(bboxes1, bboxes2, format_='cxcyczwhd'):
    """Determines the intersection over union (IoU) for two sets of
    three dimensional bounding boxes.

    Bounding boxes have to be in the format (x1, y1, z1, x2, y2, z2).

    Args:
        bboxes1: A np.array of the shape [N, 6] containing the first
            set of bounding boxes.
        bboxes2: A np.array of the shape [M, 6] containing the first
            set of bounding boxes.

    Returns:
        A tensor of shape [N, M] containing the IoU values of all 
        bounding boxes to each other.
    """
    if format_ == 'cxcyczwhd':
        bboxes1 = box_cxcyczwhd_to_xyzxyz(bboxes1)
        bboxes2 = box_cxcyczwhd_to_xyzxyz(bboxes2)

    volume_bbox1 = bboxes_volume(bboxes1)
    volume_bbox2 = bboxes_volume(bboxes2)

    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[None, :, 0])
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[None, :, 1])
    z1 = np.maximum(bboxes1[:, None, 2], bboxes2[None, :, 2])
    x2 = np.minimum(bboxes1[:, None, 3], bboxes2[None, :, 3])
    y2 = np.minimum(bboxes1[:, None, 4], bboxes2[None, :, 4])
    z2 = np.minimum(bboxes1[:, None, 5], bboxes2[None, :, 5])

    delta_x = np.clip((x2 - x1), a_min=0, a_max=None)
    delta_y = np.clip((y2 - y1), a_min=0, a_max=None)
    delta_z = np.clip((z2 - z1), a_min=0, a_max=None)

    intersection = delta_x * delta_y * delta_z
    union = volume_bbox1[:, None] + volume_bbox2 - intersection
    return intersection / union


def merge_patches(predictions, patch_positions, patch_size, image_size, mode, config=None):
    assert mode in ["average", "center_dist", "score", "volume", "volume_similarity", "custom"]
    # Initialize empty lists to store the highest scoring boxes
    boxes_dict = {}
    boxes_volume_dict = {}
    classes_dict = {}
    scores_dict = {}
    highest_scores = {}
    center_dist = {} # for center_dist mode
    min_vol_diff = {}

    if mode == "custom":
        custom_mode = True # uses center_dist for M&L organs and average for S organs
    else:
        custom_mode = False
        
    labels_small = [int(i) for i in config['labels_small'].keys()]
    labels_medium = [int(i) for i in config['labels_mid'].keys()]
    labels_large = [int(i) for i in config['labels_large'].keys()]
    
    if mode == "volume_similarity":
        median_volumes = get_median_volumes(config['bbox_properties'])

    assert patch_positions.shape[0] == len(predictions.keys())
    # Transform the normalized bounding boxes to global coordinates
    for patch_id, prediction in predictions.items():
        #print(f"processing patch {patch_id}")
        position = patch_positions[patch_id]
        boxes = prediction['pred_boxes']
        classes = prediction['pred_classes']
        scores = prediction['pred_scores']

        for box, class_, score in zip(boxes, classes, scores):
            if custom_mode:
                if class_ in labels_small:
                    mode = "average"
                elif class_ in labels_medium or class_ in labels_large:
                    mode = "center_dist"
            if mode == "average":
                skip = False
                for cord in box_cxcyczwhd_to_xyzxyz(box):
                    if cord > 0.95 or cord < 0.05:  # Skips bbox if corner point in 5% border
                        skip = True
                if skip:
                    #print("skipped bbox")
                    continue

            norm_x, norm_y, norm_z, norm_w, norm_h, norm_d = box
            local_x = norm_x * patch_size[0]
            local_y = norm_y * patch_size[1]
            local_z = norm_z * patch_size[2]
            w = norm_w * patch_size[0]
            h = norm_h * patch_size[1]
            d = norm_d * patch_size[2]
            global_x = local_x + position[0]
            global_y = local_y + position[1]
            global_z = local_z + position[2]

            # Normalize the global coordinates to the size of the image
            norm_global_box = [global_x / image_size[0], global_y / image_size[1], global_z / image_size[2], 
                                w / image_size[0], h / image_size[1], d / image_size[2]]
            
            box_volume = norm_global_box[-3] * norm_global_box[-2] * norm_global_box[-1]
            if class_ not in boxes_volume_dict.keys():
                boxes_volume_dict[class_] = 0
                highest_scores[class_] = 0
                center_dist[class_] = 99
                min_vol_diff[class_] = 99
            # If this class has not been seen before or this box has a higher score than the previous box for this class
            if mode == "volume" and box_volume > boxes_volume_dict[class_]:
                # Update the highest scoring box for this class
                #print(f"prediction for  {class_}")
                boxes_dict[class_] = norm_global_box
                classes_dict[class_]= class_
                scores_dict[class_] = score
            elif mode == "score" and score > highest_scores[class_]:
                # Update the highest scoring box for this class
                boxes_dict[class_] = norm_global_box
                classes_dict[class_]= class_
                scores_dict[class_] = score
                highest_scores[class_] = score
            elif mode == "average":
                if class_ not in boxes_dict.keys():
                    boxes_dict[class_] = []
                    scores_dict[class_] = []    
                boxes_dict[class_].append(norm_global_box)
                scores_dict[class_].append(score)
                classes_dict[class_]= class_

            elif mode == "center_dist":
                dist = np.linalg.norm((norm_x, norm_y, norm_z) - np.array([0.5, 0.5, 0.5]))
                if dist < center_dist[class_]:
                    boxes_dict[class_] = norm_global_box
                    classes_dict[class_]= class_
                    scores_dict[class_] = score
                    center_dist[class_] = dist
            elif mode == "volume_similarity" :
                vol_diff = np.abs(median_volumes[class_] - box_volume)
                if vol_diff < min_vol_diff[class_]:
                    boxes_dict[class_] = norm_global_box
                    classes_dict[class_]= class_
                    scores_dict[class_] = score
                    min_vol_diff[class_] = vol_diff
                
                
    if mode == "average" and not custom_mode:   # Average all predictions
        for class_ in boxes_dict.keys():
            boxes_dict[class_] = np.mean(boxes_dict[class_], axis=0)
            scores_dict[class_] = np.mean(scores_dict[class_])
    elif custom_mode:   # Average only predictions for small organs
        for class_ in labels_small:
            if class_ not in boxes_dict.keys():
                continue
            boxes_dict[class_] = np.mean(boxes_dict[class_], axis=0)
            scores_dict[class_] = np.mean(scores_dict[class_])


    # Convert lists to numpy arrays
    boxes_array = np.array(list(boxes_dict.values()))
    classes_array = np.array(list(classes_dict.values()))
    scores_array = np.array(list(scores_dict.values()))

    return [boxes_array], [classes_array], [scores_array]

def get_median_volumes(bbox_properties):
    median_volumes = {}
    for k,v in bbox_properties.items():
        median_volumes[int(k)] = v['median'][-3]*v['median'][-2]*v['median'][-1]
    return median_volumes
