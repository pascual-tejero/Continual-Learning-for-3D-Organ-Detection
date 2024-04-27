"""Module containing the loss functions of the transoar project."""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
# For Hausdorff
from monai.metrics import HausdorffDistanceMetric
import warnings
warnings.filterwarnings("ignore")

from transoar.utils.bboxes import generalized_bbox_iou_3d, box_cxcyczwhd_to_xyzxyz

def sigmoid_focal_loss(
    inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

class TransoarCriterion(nn.Module):
    """ This class computes the loss for TransoarNet.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, seg_proxy, seg_fg_bg, seg_msa, focal_loss=False, extra_classes=0,
                 num_classes_orig_dataset=0, config=None):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.focal_loss = focal_loss
        self._seg_proxy = seg_proxy
        self._seg_fg_bg = seg_fg_bg
        self._seg_msa = seg_msa
        self._seg_fg_bg_haus = True

        self._seg_proxy &= not self._seg_msa
        self._seg_msa &= not self._seg_proxy
        self.extra_classes = extra_classes
        self.num_classes_orig_dataset = num_classes_orig_dataset

        if seg_proxy or  seg_msa:
            self._CE = nn.CrossEntropyLoss().cuda()
            self._dice_loss = SoftDiceLoss(
                nonlin=torch.nn.Softmax(dim=1), batch_dice=True, smooth_nom=1e-05, smooth_denom=1e-05,do_bg=False
            )
            self._hausdorff_distance = HausdorffDistanceMetric(include_background=False, distance_metric="euclidean", directed=False, percentile=95.0)

        self.config = config

        # Hack to make deterministic, https://github.com/pytorch/pytorch/issues/46024
        first_component = torch.tensor([1])
        rest_components = torch.full((num_classes,), 10)
        self.cls_weights = torch.cat((first_component, rest_components)).type(torch.FloatTensor)

        if self.extra_classes > 0:
            self.cls_weights = self.cls_weights[:self.num_classes_orig_dataset + 1]

        """self.cls_weights = torch.tensor(
            [1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10]
        ).type(torch.FloatTensor)"""
    
    def prep_for_dn(self, dn_meta):
        output_known_lbs_bboxes = dn_meta["output_known_lbs_bboxes"]
        num_dn_groups, pad_size = dn_meta["num_dn_group"], dn_meta["pad_size"]
        assert pad_size % num_dn_groups == 0
        single_pad = pad_size // num_dn_groups

        return output_known_lbs_bboxes, single_pad, num_dn_groups
    
    def loss_class(self, outputs, targets, indices, num_boxes=None):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)


        if not indices:  # only for patches without any GT bboxes, (for patch-based training)
            target_classes_o = torch.tensor([], device=src_logits.device).long()
        else:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)]).long()

        target_classes = torch.full(src_logits.shape[:2], 0,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o # First tensor are the rows, second tensor are the columns

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.cls_weights.to(device=src_logits.device))
        losses = {"cls": loss_ce}

        return losses
    
    def downsampler_fn(self, img, out_size):
        """
        input sahep: B,C,H,W,D
        output sahep: B,C,H,W,D

        """
        out = img.clone()
        ratio = int(img.size()[-1] / out_size[-1])
        out = out[:,:, ::ratio, ::ratio, ::ratio]

        #return nn.functional.interpolate(img,
        #                             size=out_size,
        #                             mode='nearest',
        #                             align_corners=None,
        #                             recompute_scale_factor=None,
        #                             #antialias=False
        #)
        return out

    def loss_class_focal(self, outputs, targets, indices, num_boxes):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"]

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )

        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros(
            [src_logits.shape[0], src_logits.shape[1], src_logits.shape[2] + 1],
            dtype=src_logits.dtype,
            layout=src_logits.layout,
            device=src_logits.device,
        )
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:, :, :-1]
        loss_ce = (
            sigmoid_focal_loss(
                src_logits,
                target_classes_onehot,
                num_boxes,
                alpha=0.25,
                gamma=2,
            )
            * src_logits.shape[1]
        )
        losses = {"cls": loss_ce}
        return losses

    def loss_bboxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        if self.config["only_class_labels"]:
            losses = {}
            losses['bbox'] = torch.tensor(0).to(device=outputs['pred_logits'].device)
            losses['giou'] = torch.tensor(0).to(device=outputs['pred_logits'].device)
            return losses


        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]

        if not indices: # only for patches without any GT bboxes, (for patch-based training)
            losses = {}
            losses['bbox'] = torch.tensor(0).to(device=outputs['pred_logits'].device)
            losses['giou'] = torch.tensor(0).to(device=outputs['pred_logits'].device)
            return losses
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        loss_bbox = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_bbox_iou_3d(
            box_cxcyczwhd_to_xyzxyz(src_boxes),
            box_cxcyczwhd_to_xyzxyz(target_boxes))
        )

        loss_giou = loss_giou.sum() / num_boxes
        losses = {}
        losses['bbox'] = loss_bbox
        losses['giou'] = loss_giou
        return losses

    def loss_segmentation(self, outputs, targets):
        assert 'pred_seg' in outputs

        losses = {key:0. for key in ['segce', 'segdice'] }
        if self._seg_msa:
            assert 'neck_enc_seg' in outputs
            
            for output, output_neck in zip(outputs['pred_seg'], outputs['neck_enc_seg']):
                out_size = output.size()[2:]
                target = self.downsampler_fn(targets, out_size)

                # Get only fg and bg labels
                if self._seg_fg_bg:
                    target[target > 0] = 1
                target = target.squeeze(1).long()

                # Determine segmentation losses of MSAViT
                losses['segce'] += self._CE(output, target)
                losses['segdice'] += self._dice_loss(output, target)

                # Determine segmentation losses of neck
                losses['segce'] += self._CE(output_neck, target)
                losses['segdice'] += self._dice_loss(output_neck, target)

            return losses

        # Get only fg and bg labels
        if self._seg_fg_bg:
            targets[targets > 0] = 1
        targets = targets.squeeze(1).long()

        # Determine segmentatio losses
        loss_ce = F.cross_entropy(outputs['pred_seg'], targets)
        loss_dice = self._dice_loss(outputs['pred_seg'], targets)

        losses['segce'] = loss_ce
        losses['segdice'] = loss_dice
        return losses

    
    def hd95_loss(self, outputs, targets):
        assert 'pred_seg' in outputs

        losses = {key:0. for key in ['hd95'] }
        if self._seg_msa:
            for i, output in enumerate(outputs['pred_seg']):

                out_size = output.size()[2:]
                target = self.downsampler_fn(targets, out_size)

                # Get only fg and bg labels
                if self._seg_fg_bg:
                    target[target > 0] = 1

                # target only has foreground classes → bring into form (batch, classes, h, w, d)
                y_one_hot = torch.nn.functional.one_hot(target.long().squeeze(1), output.shape[1])
                target = y_one_hot.permute(0, 4, 1, 2, 3).float()

                # get 
                act = torch.nn.Softmax(dim=1)
                pred_segm = act(output)

                if self._seg_fg_bg_haus:
                    pred_segm = (pred_segm > 0.5)
                else:
                    raise(ValueError, "this hasn't been tested yet, please check first if it makes sense")
                    pred_segm = pred_segm.long()
                distance = self._hausdorff_distance(pred_segm, target)
                distance = torch.mean(distance)
                losses['hd95'] += distance
            
            return losses


        # Get only fg and bg labels
        if self._seg_fg_bg:
           targets[targets > 0] = 1

        # target only has foreground classes → bring into form (batch, classes, h, w, d)
        y_one_hot = torch.nn.functional.one_hot(targets.long().squeeze(1), outputs['pred_seg'].shape[1])
        targets = y_one_hot.permute(0, 4, 1, 2, 3).float()

        # get 
        act = torch.nn.Softmax(dim=1)
        pred_segm = act(outputs['pred_seg'])
        if self._seg_fg_bg:
            pred_segm = (pred_segm > 0.5)
        else:
            raise(ValueError, "this hasn't been tested yet, please check first if it makes sense")
            pred_segm = pred_segm.long()
        distance = self._hausdorff_distance(pred_segm, targets)
        distance = torch.mean(distance)
        losses = {}
        losses['hd95'] = distance
        return losses
    
    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        if self.focal_loss:
            loss_map = {
                "cls": self.loss_class_focal,
                "bbox": self.loss_bboxes
            }
        else:
            loss_map = {
                "cls": self.loss_class,
                "bbox": self.loss_bboxes
            }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        if not indices:  # handle no GT instances (for patch-based training)
            return [], []
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, seg_targets, dn_meta=None, num_epoch:int=-1,
                flag_b2_ocl_rep_mix:bool=False, flag_b1_ocl:bool=False):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Ouputs for CL has only keys: 'pred_logits', 'pred_boxes', 'pred_seg', 'neck_enc_seg'
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs' and k != 'enc_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        if flag_b1_ocl: # Case we have only class labels which batch size is 1
            pos_indices = []
        elif flag_b2_ocl_rep_mix: # Case we have replay or mixing datasets data which batch size is 2
            # Get the second sample of the batch (size=2) and the second target
            outputs_without_aux_2 = {}
            outputs_without_aux_2["pred_logits"] = outputs_without_aux["pred_logits"][1].unsqueeze(0)
            outputs_without_aux_2["pred_boxes"] = outputs_without_aux["pred_boxes"][1].unsqueeze(0)
            outputs_without_aux_2["pred_seg"] = [outputs_without_aux["pred_seg"][0][1].unsqueeze(0), 
                                                 outputs_without_aux["pred_seg"][1][1].unsqueeze(0), 
                                                 outputs_without_aux["pred_seg"][2][1].unsqueeze(0)]
            outputs_without_aux_2["neck_enc_seg"] = [outputs_without_aux["neck_enc_seg"][0][1].unsqueeze(0), 
                                                     outputs_without_aux["neck_enc_seg"][1][1].unsqueeze(0), 
                                                     outputs_without_aux["neck_enc_seg"][2][1].unsqueeze(0)]
            targets_2 = [targets[1]]
            pos_indices = self.matcher(outputs_without_aux_2, targets_2, num_epoch)

        else:
            pos_indices = self.matcher(outputs_without_aux, targets, num_epoch)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)

        # Compute losses except for contrastive losses
        losses = {}
        
        if dn_meta is not None:
            # prepare for computing denosing loss
            output_known_lbs_bboxes, single_pad, scalar = self.prep_for_dn(dn_meta)
            dn_pos_idx = []
            dn_neg_idx = []
            for i in range(len(targets)):
                if len(targets[i]["labels"]) > 0:
                    t = torch.arange(0, len(targets[i]["labels"]) - 1).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                    output_idx = output_idx.flatten()
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()

                dn_pos_idx.append((output_idx, tgt_idx))
                dn_neg_idx.append((output_idx + single_pad // 2, tgt_idx))

            for loss in ['cls', 'bbox']:
                loss_dict = self.get_loss(
                        loss,
                        output_known_lbs_bboxes,
                        targets,
                        dn_pos_idx,
                        num_boxes * scalar,
                    )
                loss_dict = {k + f"_dn": v for k, v in loss_dict.items()}
                losses.update(loss_dict)
        
        if flag_b2_ocl_rep_mix: # Case we have replay or mixing datasets data which batch size is 2
            # First compute the loss for ABDOMENCT-1K dataset which we have all targets
            for loss in ['cls', 'bbox']:
                loss_dict = self.get_loss(loss, outputs_without_aux_2, targets_2, pos_indices, num_boxes)
                losses.update(loss_dict)

            # Second compute the loss for WORD dataset which we have only class labels
            pos_indices = []
            outputs_without_aux_1 = {}
            outputs_without_aux_1["pred_logits"] = outputs["pred_logits"][0].unsqueeze(0)
            targets_1 = [targets[0]]
            
            loss_1 = self.get_loss('cls', outputs_without_aux_1, targets_1, pos_indices, num_boxes)['cls']
            losses['cls'] += loss_1

        else:
            for loss in ['cls', 'bbox']:
                loss_dict = self.get_loss(loss, outputs, targets, pos_indices, num_boxes)
                losses.update(loss_dict)

        #  seg_one2many check to block segmentation loss in one2many branch
        if (self._seg_msa or self._seg_proxy ) and not 'seg_one2many' in outputs \
            and not flag_b1_ocl and not flag_b2_ocl_rep_mix:
            loss_dict = self.loss_segmentation(outputs, seg_targets)
            losses.update(loss_dict) 
            with torch.no_grad():
                loss_dict = self.hd95_loss(outputs, seg_targets)
            losses.update(loss_dict)

        elif (self._seg_msa or self._seg_proxy ) and not 'seg_one2many' in outputs \
            and not flag_b1_ocl and flag_b2_ocl_rep_mix: 
            # For this case, seg_targets is the second target of the batch and
            # outputs is the second output of the batch
            loss_dict = self.loss_segmentation(outputs_without_aux_2, seg_targets)
            losses.update(loss_dict)
            with torch.no_grad():
                loss_dict = self.hd95_loss(outputs_without_aux_2, seg_targets)
            losses.update(loss_dict)
            
        else:
            losses['segce'] = torch.tensor(0)
            losses['segdice'] = torch.tensor(0)
            losses['hd95'] = torch.tensor(0)

        # Compute losses for the output of each intermediate layer
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets, num_epoch)
                for loss in ['cls', 'bbox']:
                    loss_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes)
                    loss_dict = {k + f"_{i}": v for k, v in loss_dict.items()}
                    losses.update(loss_dict)
                    
                if dn_meta is not None:
                    aux_outputs_known = output_known_lbs_bboxes["aux_outputs"][i]
                    loss_dict = {}
                    for loss in ['cls', 'bbox']:
                        loss_dict.update(
                            self.get_loss(
                                loss,
                                aux_outputs_known,
                                targets,
                                dn_pos_idx,
                                num_boxes * scalar,
                            )
                        )
                    loss_dict = {k + f"_{i}_dn": v for k, v in loss_dict.items()}
                    losses.update(loss_dict)
        
        if 'enc_outputs' in outputs:
            enc_outputs = outputs['enc_outputs']
            bin_targets = copy.deepcopy(targets)
            for bt in bin_targets:
                bt['labels'] = torch.zeros_like(bt['labels'])
            indices = self.matcher(enc_outputs, bin_targets, num_epoch)
            for loss in ['cls', 'bbox']:
                loss_dict = self.get_loss(loss, enc_outputs, bin_targets, indices, num_boxes)
                loss_dict = {k + f"_enc": v for k, v in loss_dict.items()}
                losses.update(loss_dict)


        return losses, pos_indices

class SoftDiceLoss(nn.Module):
    def __init__(
        self, nonlin=None, batch_dice=False, do_bg=False, 
        smooth_nom=1e-5, smooth_denom=1e-5
    ):
        super().__init__()

        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.nonlin = nonlin
        self.smooth_nom = smooth_nom
        self.smooth_denom = smooth_denom

    def forward(self, inp, target, loss_mask=None):
        shp_x = inp.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.nonlin is not None:
            inp = self.nonlin(inp)

        tp, fp, fn = get_tp_fp_fn(inp, target, axes, loss_mask, False)

        nominator = 2 * tp + self.smooth_nom
        denominator = 2 * tp + fp + fn + self.smooth_denom

        dc = nominator / denominator

        if not self.do_bg:
            if self.batch_dice:
                dc = dc[1:]
            else:
                dc = dc[:, 1:]
        dc = dc.mean()

        return 1 - dc

def get_tp_fp_fn(net_output, gt, axes=None, mask=None, square=False):
    if axes is None:
        axes = tuple(range(2, len(net_output.size())))

    shp_x = net_output.shape
    shp_y = gt.shape

    with torch.no_grad():
        if len(shp_x) != len(shp_y):
            gt = gt.view((shp_y[0], 1, *shp_y[1:]))

        if all([i == j for i, j in zip(net_output.shape, gt.shape)]):
            # if this is the case then gt is probably already a one hot encoding
            y_onehot = gt
        else:
            gt = gt.long()
            y_onehot = torch.zeros(shp_x)
            if net_output.device.type == "cuda":
                y_onehot = y_onehot.cuda(net_output.device.index)
            y_onehot.scatter_(1, gt, 1)

    tp = net_output * y_onehot
    fp = net_output * (1 - y_onehot)
    fn = (1 - net_output) * y_onehot

    if mask is not None:
        tp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(tp, dim=1)), dim=1)
        fp = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fp, dim=1)), dim=1)
        fn = torch.stack(tuple(x_i * mask[:, 0] for x_i in torch.unbind(fn, dim=1)), dim=1)

    if square:
        tp = tp ** 2
        fp = fp ** 2
        fn = fn ** 2

    tp = tp.sum(dim=axes, keepdim=False)
    fp = fp.sum(dim=axes, keepdim=False)
    fn = fn.sum(dim=axes, keepdim=False)
    return tp, fp, fn