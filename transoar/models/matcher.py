"""Module containing the hungarian matcher, adapted from https://github.com/facebookresearch/detr."""

import math
import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
import numpy as np

from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz, generalized_bbox_iou_3d


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, 
                 dense_matching: bool = False, dense_matching_lambda: float = 0.5,
                 class_matching: bool = False, class_matching_query_split: list = [],
                 recursive_dm_dn = False, extra_classes: int = 0, num_classes_orig_dataset: int = 0,
                 config=None):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs can't be 0"

        self.dense_matching=dense_matching
        self.dense_matching_lambda=dense_matching_lambda

        self.class_matching = class_matching 
        self.query_split = class_matching_query_split
        self.recursive_dm_dn = recursive_dm_dn

        self.extra_classes = extra_classes
        self.num_classes_orig_dataset = num_classes_orig_dataset
        self.config = config
        #assert (dense_matching != class_matching) or (not dense_matching and not class_matching) , "class matching in combination with dense matching not implemented yet"
        """#TODO add category wise matching
        self.classes_s = [4]
        self.classes_m = [2,3,5]
        self.classes_l = [1]"""

    @torch.no_grad()
    def forward(self, outputs, targets, num_epoch: int=-1):
        """ Performs the matching
        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 6] with the predicted box coordinates
            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        if self.recursive_dm_dn:
            dm_flag = num_epoch>0 and num_epoch%2 != 0 
        else:
            dm_flag = True
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 6]

       
        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])

        if tgt_ids.nelement() == 0: # if no targets (for patch-based training)
            return []
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids.long()]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)

        cost_giou = -generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(out_bbox),
            box_cxcyczwhd_to_xyzxyz(tgt_bbox)
        )

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()
        C = C.nan_to_num()

        sizes = [len(v["boxes"]) for v in targets]

        if dm_flag and self.dense_matching and not self.class_matching:
            indices = []
            for i, c in enumerate(C.split(sizes, -1)):
                k = c[i].shape[-1] # classes=instances in GT
                if k == 0: # for patch based if no GT
                    repeats = 0
                    c_for_matching = c[i].repeat(1, repeats) # repeat GT
                else:
                    if self.extra_classes > 0 or self.config["CL_replay"] or self.config["mixing_datasets"]:
                        c_for_matching = c[i]
                    else:
                        repeats = math.ceil(self.dense_matching_lambda * num_queries / k)
                        c_for_matching = c[i].repeat(1, repeats) # repeat GT
                idx_logits, idx_classes = linear_sum_assignment(c_for_matching)
                idx_classes = idx_classes % sizes[i] # modulo num_classes (sizes[i]) to get class_ids from matched ids
                indices.append((idx_logits, idx_classes))

            ret = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return ret
        elif dm_flag and self.class_matching:
            indices = []
            for i, c in enumerate(C.split(sizes, -1)): # iterate batch
                k = c[i].shape[-1]
                
                c_query_groups = torch.split(c[i], self.query_split, dim=0) #split cost tensor according to query groups
                offset = 0  
                concat_idx_logits = []
                concat_idx_classes = []
                current_targets = targets[i]["labels"] -1 # deduct one from class ids for matching
                current_targets = current_targets.tolist()
                is_sorted = all(a <= b for a, b in zip(current_targets, current_targets[1:]))
                assert is_sorted

                # enumerate is needed to handle cases with missing labels → column in C is missing
                # it is assumed that the target labels are ascending
                cost_class_idx = 0  # needed to handle missing labeles
                for j, q_group in enumerate(c_query_groups):
                    if j not in current_targets:
                        offset += q_group.shape[0]
                        continue 
                    if self.dense_matching:
                        num_matches = int(self.dense_matching_lambda * q_group[:,cost_class_idx].shape[-1])
                        idx_logits = torch.topk(q_group[:,cost_class_idx], num_matches , largest=False)[-1]
                    else:
                        idx_logits = np.argmin(q_group[:,cost_class_idx]) # finds optimum for splitted queries and corresponding class
                    # add id offset caused by splitting
                    idx_logits += offset
                    offset += q_group.shape[0]
                    # concat the split matches again, so they can be processed by the existing criterion
                    concat_idx_logits.append(idx_logits)
                    if self.dense_matching:
                        concat_idx_classes.append(torch.tensor([cost_class_idx]).repeat(idx_logits.shape[0]))
                    else:
                        concat_idx_classes.append(cost_class_idx)
                    cost_class_idx += 1 # only increments if valid indices were added for current cost_class_idx
                if self.dense_matching:
                    concat_idx_logits = torch.cat(concat_idx_logits, dim=0)
                    concat_idx_classes = torch.cat(concat_idx_classes, dim=0)
                indices.append((concat_idx_logits, concat_idx_classes))
            ret = [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
            return ret
        else:
            indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
            return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]