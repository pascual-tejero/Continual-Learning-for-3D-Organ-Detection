"""Module containing the trainer of the transoar project."""

import copy
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import PIL.Image
from tqdm import tqdm
import numpy as np
from transoar.evaluator import DetectionEvaluator
from transoar.inference import inference
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
import io

# helper function: generate box_plot of grads in tensorboard
def gen_box_plot(grads_list, num_epoch_list, name=None):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.boxplot(grads_list)
    plt.title(f'{name} of from epoch {num_epoch_list[0]} to epoch {num_epoch_list[-1]}')
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    plt.xticks(np.arange(len(grads_list))+1, num_epoch_list)
    buf.seek(0)
    return buf

class Trainer:

    def __init__(
        self, train_loader, val_loader, model, criterion, optimizer, scheduler,
        device, config, path_to_run, epoch, metric_start_val, dense_hybrid_criterion
    ):
        self._train_loader = train_loader
        self._val_loader = val_loader
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._scheduler = scheduler
        self._device = device
        self._path_to_run = path_to_run
        self._epoch_to_start = epoch
        self._config = config
        self.log_grad = config.get('log_grad', False)
        self.log_grad_every_epoch = config.get('log_grad_every_epoch', 10)
        self._hybrid = config.get('hybrid_matching', False)
        self._hybrid_K = config.get('hybrid_K', 0)
        self._dense_hybrid_criterion = dense_hybrid_criterion
        
        if self.log_grad:
            self.log_grads_list_pos = []
            self.log_grads_list_neg = []
            self.log_epoch_list = []

        self._writer = SummaryWriter(log_dir=path_to_run)
        self._scaler = GradScaler()
        
        self._evaluator = DetectionEvaluator(
            classes=list(config['labels'].values()),
            classes_small=config['labels_small'],
            classes_mid=config['labels_mid'],
            classes_large=config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=True
        )

        # Init main metric for checkpoint
        self._main_metric_key = 'mAP_coco'
        self._main_metric_max_val = metric_start_val
    
    def _train_one_epoch(self, num_epoch):
        self._model.train()
        # self._criterion.train()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        loss_dn_agg = {}
        # Aux loss
        loss_aux_agg = {}
        loss_aux_agg_dn = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # contrastive loss
        loss_contrast_agg = 0
        # hybrid matching
        loss_bbox_one2many_agg = 0
        loss_giou_one2many_agg = 0
        loss_cls_one2many_agg = 0
        loss_seg_ce_one2many_agg = 0
        loss_seg_dice_one2many_agg = 0
        
        pos_query_grads_list = torch.Tensor([])
        neg_query_grads_list = torch.Tensor([])

        progress_bar = tqdm(self._train_loader)
        for data, _, bboxes, seg_targets in progress_bar:
            # Put data to gpu
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
        
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                det_targets.append(target)
            # print('det_targets: ', det_targets) 
            # quit()

            # Make prediction
            with autocast(): 
                # print(det_targets)
                # # Change from cx,cy,cz,w,h,d to x1,y1,z1,x2,y2,z2
                # for target in det_targets:
                #     target['boxes'] = torch.cat((target['boxes'][:, :3] - target['boxes'][:, 3:]/2, 
                #                                 target['boxes'][:, :3] + target['boxes'][:, 3:]/2), dim=1)
                # print("=====================================")
                # print('det_targets: ', det_targets)
                # quit()
                out, contrast_losses, dn_meta = self._model(data, det_targets, num_epoch=num_epoch)
                loss_dict, pos_indices = self._criterion(out, det_targets, seg_targets, dn_meta, num_epoch=num_epoch)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                if self._hybrid: # hybrid matching
                    outputs_one2many = dict()
                    outputs_one2many["pred_logits"] = out["pred_logits_one2many"]
                    outputs_one2many["pred_boxes"] = out["pred_boxes_one2many"]
                    outputs_one2many["aux_outputs"] = out["aux_outputs_one2many"]
                    outputs_one2many["seg_one2many"] = True
                    if self._dense_hybrid_criterion: # DM in additional branch
                        loss_dict_one2many, _ = self._dense_hybrid_criterion(outputs_one2many, det_targets, seg_targets) # det_targets replaces det_many_targets
                    else:  # regular one-to-many branch
                        det_many_targets = copy.deepcopy(det_targets)
                        # repeat the targets
                        for target in det_many_targets:
                            target["boxes"] = target["boxes"].repeat(self._hybrid_K, 1)
                            target["labels"] = target["labels"].repeat(self._hybrid_K)

                        loss_dict_one2many, _ = self._criterion(outputs_one2many, det_many_targets, seg_targets)
                    del loss_dict_one2many['hd95']
                    for key, value in loss_dict_one2many.items():
                        if key + "_one2many" in loss_dict.keys():
                            loss_dict[key + "_one2many"] += value * self._config['hybrid_loss_weight_one2many']
                        else:
                            loss_dict[key + "_one2many"] = value * self._config['hybrid_loss_weight_one2many']


                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]
                for loss_key, loss_val in contrast_losses.items():
                    loss_abs += loss_val # already multiplied coefficient in transoarnet.py
                    loss_contrast_agg += loss_val 

            self._optimizer.zero_grad()
            self._scaler.scale(loss_abs).backward()
            
            # Clip grads to counter exploding grads
            max_norm = self._config['clip_max_norm']
            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm)

            self._scaler.step(self._optimizer)
            self._scaler.update()
            
            # log gradients of positive & negative queries
            if self.log_grad:
                for name, param in self._model.named_parameters():
                    if name == 'query_embed.weight':
                        _, tgt_param_grad = torch.split(param.grad, param.size(1)//2, dim=1) # only fetch grads of tgt
                        #tgt_param_grad = param.grad
                        neg_indices = param.new_ones(param.size(0)).bool()

                        neg_indices[pos_indices[0][0]] = False
                        pos_query_grads = tgt_param_grad[pos_indices[0][0]]
                        neg_query_grads = tgt_param_grad[neg_indices]
                        # remove nan
                        pos_query_grads[torch.isnan(pos_query_grads)] = torch.tensor(0.0)
                        neg_query_grads[torch.isnan(neg_query_grads)] = torch.tensor(0.0)
                        pos_query_grads_list = torch.cat((pos_query_grads_list.cuda(), pos_query_grads), dim=0)
                        neg_query_grads_list = torch.cat((neg_query_grads_list.cuda(), neg_query_grads), dim=0)

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()
            if self._hybrid: # hybrid matching
                loss_bbox_one2many_agg += loss_dict['bbox_one2many'].item()
                loss_giou_one2many_agg += loss_dict['giou_one2many'].item()
                loss_cls_one2many_agg += loss_dict['cls_one2many'].item()
                loss_seg_ce_one2many_agg += loss_dict['segce_one2many'].item()
                loss_seg_dice_one2many_agg += loss_dict['segdice_one2many'].item()

            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95
            
            if dn_meta is not None:
                if len(loss_dn_agg) == 0: # initialize loss entries
                    loss_dn_agg['bbox_dn'] = 0
                    loss_dn_agg['giou_dn'] = 0
                    loss_dn_agg['cls_dn'] = 0
                else:
                    loss_dn_agg['bbox_dn'] += loss_dict[f'bbox_dn'].item()
                    loss_dn_agg['giou_dn'] += loss_dict[f'giou_dn'].item()
                    loss_dn_agg['cls_dn'] += loss_dict[f'cls_dn'].item()
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
                if dn_meta is not None:
                    if len(loss_aux_agg_dn) == 0: # initialize loss entries
                        for i in range(len(out["aux_outputs"])):
                            loss_aux_agg_dn[f'bbox_{i}_dn'] = 0
                            loss_aux_agg_dn[f'giou_{i}_dn'] = 0
                            loss_aux_agg_dn[f'cls_{i}_dn'] = 0
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg_dn[f'bbox_{i}_dn'] += loss_dict[f'bbox_{i}_dn'].item()
                        loss_aux_agg_dn[f'giou_{i}_dn'] += loss_dict[f'giou_{i}_dn'].item()
                        loss_aux_agg_dn[f'cls_{i}_dn'] += loss_dict[f'cls_{i}_dn'].item()
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
            
        loss = loss_agg / len(self._train_loader)
        #print(f'total train loss for epoch {num_epoch}: '+str(loss))
        loss_bbox = loss_bbox_agg / len(self._train_loader)
        loss_giou = loss_giou_agg / len(self._train_loader)
        loss_cls = loss_cls_agg / len(self._train_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._train_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._train_loader)
        if self._hybrid: # hybrid matching
            loss_bbox_one2many = loss_bbox_one2many_agg / len(self._train_loader)
            loss_giou_one2many = loss_giou_one2many_agg / len(self._train_loader)
            loss_cls_one2many = loss_cls_one2many_agg / len(self._train_loader)
            loss_seg_ce_one2many = loss_seg_ce_one2many_agg / len(self._train_loader)
            loss_seg_dice_one2many = loss_seg_dice_one2many_agg / len(self._train_loader)
        
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._train_loader)
        else:
            seg_hd95 = 0

        loss_contrast = loss_contrast_agg / len(self._train_loader)
        
        if len(loss_dn_agg) != 0:
            for key in loss_dn_agg:
                value = loss_dn_agg[key] / len(self._train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._train_loader)
                self._writer.add_scalar("train_aux/"+key, value, num_epoch)
        if len(loss_aux_agg_dn) != 0:
            for key in loss_aux_agg_dn:
                value = loss_aux_agg_dn[key] / len(self._train_loader)
                self._writer.add_scalar("dn/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("train_enc/bbox_enc", loss_enc_bbox_agg/len(self._train_loader), num_epoch)
            self._writer.add_scalar("train_enc/giou_enc", loss_enc_giou_agg/len(self._train_loader), num_epoch)
            self._writer.add_scalar("train_enc/cls_enc", loss_enc_cls_agg/len(self._train_loader), num_epoch)

        if self._hybrid: # log many2one just if hybrid matching is activated
            self._write_to_logger(
                    num_epoch, 'train', 
                    total_loss=loss,
                    bbox_loss=loss_bbox,
                    giou_loss=loss_giou,
                    cls_loss=loss_cls,
                    seg_ce_loss=loss_seg_ce,
                    seg_dice_loss=loss_seg_dice,
                    seg_hd95=seg_hd95, # log Hausdorff
                    bbox_loss_one2many = loss_bbox_one2many,
                    giou_loss_one2many = loss_giou_one2many,
                    cls_loss_one2many = loss_cls_one2many,
                    seg_ce_loss_one2many = loss_seg_ce_one2many,
                    seg_dice_loss_one2many = loss_seg_dice_one2many
            )
        else:
            self._write_to_logger(
                num_epoch, 'train', 
                total_loss=loss,
                bbox_loss=loss_bbox,
                giou_loss=loss_giou,
                cls_loss=loss_cls,
                seg_ce_loss=loss_seg_ce,
                seg_dice_loss=loss_seg_dice,
                seg_hd95=seg_hd95, # log Hausdorff
                contrast_loss=loss_contrast,
            )
            
            if self.log_grad and num_epoch % self.log_grad_every_epoch == 0:
                self.log_grads_list_pos.append(torch.flatten(pos_query_grads_list, 0).cpu())
                self.log_grads_list_neg.append(torch.flatten(neg_query_grads_list, 0).cpu())
                self.log_epoch_list.append(num_epoch)
                
                avg_pos_queries_grad = torch.abs(pos_query_grads_list).mean()
                avg_neg_queries_grad = torch.abs(neg_query_grads_list).mean()
                self._writer.add_scalar("grads/avg_pos_queries_grad", avg_pos_queries_grad, num_epoch)
                self._writer.add_scalar("grads/avg_neg_queries_grad", avg_neg_queries_grad, num_epoch)
                
                
                for name, param in self._model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._writer.add_histogram('grads/' + name, param.grad, int(num_epoch))
                

    @torch.no_grad()
    def _validate(self, num_epoch):
        
        self._model.eval()
        # self._criterion.eval()

        loss_agg = 0
        loss_bbox_agg = 0
        loss_giou_agg = 0
        loss_cls_agg = 0
        loss_seg_ce_agg = 0
        loss_seg_dice_agg = 0
        # log Hausdorff
        hd95_agg = 0
        # Aux loss
        loss_aux_agg = {}
        # Two stage
        loss_enc_bbox_agg = 0
        loss_enc_giou_agg = 0
        loss_enc_cls_agg = 0
        progress_bar = tqdm(self._val_loader)
        for data, _, bboxes, seg_targets in progress_bar:
            # Put data to gpu
            data, seg_targets = data.to(device=self._device), seg_targets.to(device=self._device)
        
            det_targets = []
            for item in bboxes:
                target = {
                    'boxes': item[0].to(dtype=torch.float, device=self._device),
                    'labels': item[1].to(device=self._device)
                }
                det_targets.append(target)

            # Make prediction
            with autocast():
                out = self._model(data)
                loss_dict, _ = self._criterion(out, det_targets, seg_targets)

                if self._criterion._seg_proxy: # log Hausdorff
                    hd95 = loss_dict['hd95'].item()
                del loss_dict['hd95'] # remove HD from loss, so it does not influence loss_abs

                # Create absolute loss and mult with loss coefficient
                loss_abs = 0
                for loss_key, loss_val in loss_dict.items():
                    loss_abs += loss_val * self._config['loss_coefs'][loss_key.split('_')[0]]

            # Evaluate validation predictions based on metric
            pred_boxes, pred_classes, pred_scores = inference(out)
            self._evaluator.add(
                pred_boxes=pred_boxes,
                pred_classes=pred_classes,
                pred_scores=pred_scores,
                gt_boxes=[target['boxes'].detach().cpu().numpy() for target in det_targets],
                gt_classes=[target['labels'].detach().cpu().numpy() for target in det_targets]
            )

            loss_agg += loss_abs.item()
            loss_bbox_agg += loss_dict['bbox'].item()
            loss_giou_agg += loss_dict['giou'].item()
            loss_cls_agg += loss_dict['cls'].item()
            loss_seg_ce_agg += loss_dict['segce'].item()
            loss_seg_dice_agg += loss_dict['segdice'].item()
            if self._criterion._seg_proxy: # log Hausdorff
                hd95_agg += hd95

            
            if "aux_outputs" in out:
                if len(loss_aux_agg) == 0: # initialize loss entries
                    for i in range(len(out["aux_outputs"])):
                        loss_aux_agg[f'bbox_{i}'] = 0
                        loss_aux_agg[f'giou_{i}'] = 0
                        loss_aux_agg[f'cls_{i}'] = 0
                for i in range(len(out["aux_outputs"])):
                    loss_aux_agg[f'bbox_{i}'] += loss_dict[f'bbox_{i}'].item()
                    loss_aux_agg[f'giou_{i}'] += loss_dict[f'giou_{i}'].item()
                    loss_aux_agg[f'cls_{i}'] += loss_dict[f'cls_{i}'].item()
                
            if "enc_outputs" in out:
                loss_enc_bbox_agg += loss_dict['bbox_enc'].item()
                loss_enc_giou_agg += loss_dict['giou_enc'].item()
                loss_enc_cls_agg += loss_dict['cls_enc'].item()
            memory_allocated, memory_cached = get_gpu_memory(self._device)
            progress_bar.set_postfix({'cached': "{:.2f}GB".format(memory_cached/(1024**3))})
                
        loss = loss_agg / len(self._val_loader)
        loss_bbox = loss_bbox_agg / len(self._val_loader)
        loss_giou = loss_giou_agg / len(self._val_loader)
        loss_cls = loss_cls_agg / len(self._val_loader)
        loss_seg_ce = loss_seg_ce_agg / len(self._val_loader)
        loss_seg_dice = loss_seg_dice_agg / len(self._val_loader)
        if self._criterion._seg_proxy:  # log Hausdorff
            seg_hd95 = hd95_agg / len(self._val_loader)
        else:
            seg_hd95 = 0

        metric_scores = self._evaluator.eval()
        self._evaluator.reset()

        # Check if new best checkpoint
        if metric_scores[self._main_metric_key] >= self._main_metric_max_val \
            and not self._config['debug_mode']:
            self._main_metric_max_val = metric_scores[self._main_metric_key]
            self._save_checkpoint(
                num_epoch,
                f'model_best_{metric_scores[self._main_metric_key]:.3f}_in_ep{num_epoch}.pt'
            )
            
        if len(loss_aux_agg) != 0:
            for key in loss_aux_agg:
                value = loss_aux_agg[key] / len(self._val_loader)
                self._writer.add_scalar("val_aux/"+key, value, num_epoch)
        if loss_enc_bbox_agg or loss_enc_giou_agg or loss_enc_cls_agg:
            self._writer.add_scalar("val_enc/bbox_enc", loss_enc_bbox_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/giou_enc", loss_enc_giou_agg/len(self._val_loader), num_epoch)
            self._writer.add_scalar("val_enc/cls_enc", loss_enc_cls_agg/len(self._val_loader), num_epoch)

        # Write to logger
        self._write_to_logger(
            num_epoch, 'val', 
            total_loss=loss,
            bbox_loss=loss_bbox,
            giou_loss=loss_giou,
            cls_loss=loss_cls,
            seg_ce_loss=loss_seg_ce,
            seg_dice_loss=loss_seg_dice
        )

        self._write_to_logger(
            num_epoch, 'val_metric',
            mAPcoco=metric_scores['mAP_coco'],
            mAPcocoS=metric_scores['mAP_coco_s'],
            mAPcocoM=metric_scores['mAP_coco_m'],
            mAPcocoL=metric_scores['mAP_coco_l'],
            mAPnndet=metric_scores['mAP_nndet'],
            AP10=metric_scores['AP_IoU_0.10'],
            AP50=metric_scores['AP_IoU_0.50'],
            AP75=metric_scores['AP_IoU_0.75'],
            seg_hd95=seg_hd95 # log Hausdorff
        )
        
        

    def run(self):
        if self._epoch_to_start == 0:   # For initial performance estimation
            self._validate(0)
        
        for epoch in range(self._epoch_to_start + 1, self._config['epochs'] + 1):
            print("starting epoch ", epoch)
            self._train_one_epoch(epoch)

            # Log learning rates
            self._write_to_logger(
                epoch, 'lr',
                backbone=self._optimizer.param_groups[0]['lr'],
                neck=self._optimizer.param_groups[1]['lr']
            )
            
            if epoch % 50 == 0 and self.log_grad:
                print('logging grads boxplot...')
                plot_buf_pos = gen_box_plot(self.log_grads_list_pos, self.log_epoch_list, 'pos queries')
                plot_buf_neg = gen_box_plot(self.log_grads_list_neg, self.log_epoch_list, 'neg queries')
                image_pos, image_neg = PIL.Image.open(plot_buf_pos), PIL.Image.open(plot_buf_neg)
                image_pos, image_neg = ToTensor()(image_pos), ToTensor()(image_neg)
                self._writer.add_image('boxplot of pos queries grads', image_pos, epoch)
                self._writer.add_image('boxplot of neg queries grads', image_neg, epoch)

            if epoch % self._config['val_interval'] == 0:
                self._validate(epoch)

            self._scheduler.step()

            if not self._config['debug_mode']:
                self._save_checkpoint(epoch, 'model_last.pt')
            # fixed checkpoints at each 200 epochs:
            if (epoch % 500) == 0:
                self._save_checkpoint(epoch, f'model_epoch_{epoch}.pt')

    def _write_to_logger(self, num_epoch, category, **kwargs):
        for key, value in kwargs.items():
            name = category + '/' + key
            self._writer.add_scalar(name, value, num_epoch)

    def _save_checkpoint(self, num_epoch, name):
        # Delete prior best checkpoint
        if 'best' in name:
            [path.unlink() for path in self._path_to_run.iterdir() if 'best' in str(path)]

        torch.save({
            'epoch': num_epoch,
            'metric_max_val': self._main_metric_max_val,
            'model_state_dict': self._model.state_dict(),
            'optimizer_state_dict': self._optimizer.state_dict(),
            'scheduler_state_dict': self._scheduler.state_dict(),
        }, self._path_to_run / name)

def get_gpu_memory(device):
    #torch.cuda.empty_cache()
    memory_allocated = torch.cuda.memory_allocated(device)
    memory_cached = torch.cuda.memory_cached(device)
    return memory_allocated, memory_cached
