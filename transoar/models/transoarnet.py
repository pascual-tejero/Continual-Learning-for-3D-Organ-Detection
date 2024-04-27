"""Main model of the transoar project."""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import os, re

from transoar.models.build import build_backbone, build_neck, build_pos_enc
from transoar.models.necks.def_detr_transformer import inverse_sigmoid
from transoar.models.necks.cdn import dn_post_process, prepare_for_cdn, prepare_for_dn
from transoar.models.matcher import HungarianMatcher
from transoar.utils.io import load_json

class TransoarNet(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.hybrid = config.get('hybrid_matching', False)
        
        if self.hybrid:
            self.num_queries_one2one = config['neck']['num_queries']

            self.hidden_dim = config['neck']['hidden_dim']
            self.num_queries = self.num_queries_one2one + config['hybrid_T']
        else:
            self.hidden_dim = config['neck']['hidden_dim']
            self.num_queries = config['neck']['num_queries']

        # load data info
        data_path = os.environ.get('TRANSOAR_DATA')
        data_dir = Path(data_path).resolve()
        data_config = load_json(data_dir / config['dataset'] / "data_info.json")
        self.num_classes = len(data_config['labels'])
        self.extra_classes = config["backbone"]["num_organs"] - self.num_classes

        if self.extra_classes > 0:
            self.num_classes = config["backbone"]["num_organs"]
            self.num_classes_orig_dataset = len(data_config['labels'])

        config['neck']['num_classes'] = self.num_classes
        self._input_level = config['neck']['input_level']

        # two stage & DAB, disabled if not configured
        self.use_dab = config.get('neck', {}).get('use_dab', False)
        self.num_patterns = config.get('neck', {}).get('num_patterns', 0) # DAB DETR uses num_pattern=3
        self.two_stage = config.get('neck', {}).get('two_stage', False)
        self.box_refine = config.get('neck', {}).get('box_refine', False)
        self.device = config['device']
        self.is_contrastive = config.get('neck', {}).get('contrastive', {}).get('enabled', False)

        # Use auxiliary decoding losses if required
        self._aux_loss = config['neck']['aux_loss']

        # Get backbone
        self._backbone = build_backbone(config['backbone'])
        self._backbone_name = config['backbone']['name']

        # Get neck
        self._neck = build_neck(config['neck'])

        # Get heads
        self._cls_head = nn.Linear(self.hidden_dim, self.num_classes + 1)
        self._bbox_reg_head = MLP(self.hidden_dim, self.hidden_dim, 6, 3)
        
        # Hungarian matcher for positive samples in contrastive learning
        if self.is_contrastive:
            self.matcher = HungarianMatcher(
                cost_class=config['set_cost_class'],
                cost_bbox=config['set_cost_bbox'],
                cost_giou=config['set_cost_giou']
                )
        
        # Get projections and embeddings
        if not self.two_stage:
            if not self.use_dab:
                self.query_embed = nn.Embedding(self.num_queries, self.hidden_dim*2) # 2 -> tgt + query_pos
            else:
                self.tgt_embed = nn.Embedding(self.num_queries, self.hidden_dim)
                self.refpoint_embed = nn.Embedding(self.num_queries, 6)
                """if random_refpoints_xy: # convert to 3d, disabled by default in DAB DETR
                    # import ipdb; ipdb.set_trace()
                    self.refpoint_embed.weight.data[:, :2].uniform_(0,1)
                    self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(self.refpoint_embed.weight.data[:, :2])
                    self.refpoint_embed.weight.data[:, :2].requires_grad = False"""
        if self.num_patterns > 0:
            self.patterns_embed = nn.Embedding(self.num_patterns, self.hidden_dim)

        # Get positional encoding
        self._pos_enc = build_pos_enc(config['neck'])

        self._reset_parameter()

        self._seg_proxy = config['backbone']['use_seg_proxy_loss']
        self._msa_seg = config['backbone'].get('use_msa_seg_loss', False)

        self._seg_proxy &= not self._msa_seg
        self._msa_seg &= not self._seg_proxy


        if self._seg_proxy:
            in_channels = config['backbone']['start_channels']
            out_channels = 2 if config['backbone']['fg_bg'] else config['neck']['num_organs'] + 1 # inc background
            self._seg_head = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
        

        if self._msa_seg: ################# MSA Segmentation
            in_channels = config['backbone']['fpn_channels']
            out_channels =  2 if config['backbone']['fg_bg'] else config['backbone']['num_organs'] + 1 
            self._seg_neck = nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=1)
    
        
        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (self._neck.decoder.num_layers + 1) if self.two_stage else self._neck.decoder.num_layers
        if self.box_refine:
            self._cls_head = _get_clones(self._cls_head, num_pred)
            self._bbox_reg_head = _get_clones(self._bbox_reg_head, num_pred)
            nn.init.constant_(self._bbox_reg_head[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self._neck.decoder.bbox_embed = self._bbox_reg_head
            self._neck.decoder.class_embed = self._cls_head

        if self.two_stage:
            # two stage should run with box_refine enabled!
            assert self.box_refine is True
            # hack implementation for two-stage
            self._neck.decoder.class_embed = self._cls_head
            for box_embed in self._bbox_reg_head:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        if self.is_contrastive:
            # contrastive projector
            contras_dim = config['neck']['contrastive']['dim']
            self.eqco = config['neck']['contrastive']['eqco']
            self.tau = config['neck']['contrastive']['tau']
            self.contras_loss_coeff = config['neck']['contrastive']['loss_coeff']
            self.projector = nn.Sequential(
                nn.Linear(self.num_classes+7, contras_dim), # num_classes + 1 + 6 (box size)
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.predictor = nn.Sequential(
                nn.Linear(contras_dim, contras_dim),
                nn.ReLU(),
                nn.Linear(contras_dim, contras_dim),
            )
            self.similarity_f = nn.CosineSimilarity(dim=2)
        
        # denoising config
        self.dn = config.get('neck', {}).get('dn', {'enabled': False,
                                                    'type': 'cdn',
                                                    'multiscale': False,
                                                    'dn_number': 0,  # number of dn groups
                                                    'dn_box_noise_ratio': 0,
                                                    'multiscale_box_noise_ratio_max': 0.,
                                                    'dn_label_noise_ratio': 0,
                                                    'multiscale_label_noise_ratio_max': 0,
                                                    })
        if self.dn['enabled']:
            self.label_enc = nn.Embedding(self.num_classes + 1, self.hidden_dim)
        if self.is_contrastive:
            # query contrast should run with contrastive denoising enabled!
            assert self.dn['enabled'] is True and self.dn['type'] == 'cdn'
        assert not (self.hybrid and self.dn['enabled']), "incompatible matching modes enabled"
        assert not (self.hybrid and config.get('dense_matching', False)), "incompatible matching modes enabled"
        assert not (self.hybrid and self.is_contrastive), "incompatible matching modes enabled"
        
    def _reset_parameter(self):
        nn.init.constant_(self._bbox_reg_head.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data, 0)
        nn.init.constant_(self._bbox_reg_head.layers[-1].bias.data[2:], -2.0)


    def forward(self, x, targets=None, num_epoch: int=-1): # in trainer, None when !self.training
        # targets: list of dict{'boxes': , 'labels':}

        out_backbone = self._backbone(x)

        # Retrieve fmaps
        det_srcs = []
        for key, value in out_backbone.items():
            obj_ = re.sub('[^a-zA-Z]+', '', key)     
            if obj_ in ['P']: # msa or def detr
                if int(key[-1]) < int(self._input_level[-1]):
                    continue
                else:
                    if self._backbone_name == 'swin_unetr':
                        B, C, H, W, D = value.shape
                        det_srcs.append(value.view(B,self.hidden_dim, -1, W, D))
                    else:
                        det_srcs.append(value)

            elif obj_ in ['res']: # resnet3D
                if int(key[-1]) < int(self._input_level[-1]):
                    continue
                else:
                    det_srcs.append(value)
        
        det_masks = []
        det_pos = []
        for idx, fmap in enumerate(det_srcs):
            det_srcs[idx] = fmap
            mask_ = torch.zeros_like(fmap[:, 0], dtype=torch.bool)    # No mask needed
            pos_ = self._pos_enc(fmap)

            det_masks.append(mask_)
            det_pos.append(pos_)

        if self.hybrid and not self.training:
            save_num_queries = self.num_queries
            self.num_queries = self.num_queries_one2one

        if self.two_stage:
            query_embeds = None
        elif self.use_dab:
            if self.num_patterns == 0:
                tgt_embed = self.tgt_embed.weight           # nq, 256
                refanchor = self.refpoint_embed.weight      # nq, 6
                query_embeds = torch.cat((tgt_embed, refanchor), dim=1)
            else:
                # multi patterns
                tgt_embed = self.tgt_embed.weight           # nq, 256
                pat_embed = self.patterns_embed.weight      # num_pat, 256
                tgt_embed = tgt_embed.repeat(self.num_patterns, 1) # nq*num_pat, 256
                pat_embed = pat_embed[:, None, :].repeat(1, self.num_queries, 1).flatten(0, 1) # nq*num_pat, 256
                tgt_all_embed = tgt_embed + pat_embed
                refanchor = self.refpoint_embed.weight.repeat(self.num_patterns, 1)      # nq*num_pat, 6
                query_embeds = torch.cat((tgt_all_embed, refanchor), dim=1)
        elif self.hybrid: 
            query_embeds = self.query_embed.weight[0 : self.num_queries, :]
        else:
            query_embeds = self.query_embed.weight

        

        if self.training and self.dn['enabled'] and self.dn['dn_number'] > 0 and num_epoch%2 ==0:
            if self.dn['type'] == 'cdn':
                dn_func = prepare_for_cdn
            elif self.dn['type'] == 'dn':
                dn_func = prepare_for_dn
            else:
                raise NotImplementedError('not implemented dn type!')
            input_query_label, input_query_bbox, dn_mask, dn_meta = dn_func(
                dn_args=(targets, 
                    self.dn['dn_number'], 
                    self.dn['multiscale'],
                    self.dn['dn_label_noise_ratio'], 
                    self.dn['multiscale_label_noise_ratio_max'],
                    self.dn['dn_box_noise_ratio'], 
                    self.dn['multiscale_box_noise_ratio_max'],
                    ), 
                training=self.training,
                is_contrastive=self.is_contrastive,
                num_queries=self.num_queries,
                num_classes=self.num_classes,
                hidden_dim=self.hidden_dim,
                label_enc=self.label_enc,
            )
            

                                                    


        else:
            input_query_bbox  = None
            input_query_label = None
            dn_mask = None
            dn_meta = None
        
        if self.hybrid:
            dn_mask = (torch.zeros([self.num_queries,self.num_queries,]).bool().to(det_srcs[-1].device)) # attn mask to limit attn to O2O
            dn_mask[self.num_queries_one2one :,0 : self.num_queries_one2one,] = True
            dn_mask[0 : self.num_queries_one2one,self.num_queries_one2one :,] = True

        out_neck = self._neck(   # [Batch, Queries, HiddenDim]         
            det_srcs,
            det_masks,
            query_embeds,
            det_pos,
            dn_mask,
            input_query_bbox,
            input_query_label,
            targets,
        )


        # Relative offset box and logit prediction
        hs, init_reference_out, inter_references_out = out_neck['hs'], out_neck['init_reference_out'], out_neck['inter_references_out']
        spatial_shapes, memory = out_neck['spatial_shapes'], out_neck['memory']

        if self.two_stage:
            enc_outputs_class, enc_outputs_coord_unact = out_neck['enc_outputs_class'], out_neck['enc_outputs_coord_unact']


        if self.hybrid and self.training:
            outputs_classes_one2many = []
            outputs_coords_one2many = []
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference_out
            else:
                reference = inter_references_out[lvl - 1]
            reference = inverse_sigmoid(reference)

            if self.box_refine:
                outputs_class = self._cls_head[lvl](hs[lvl])
                tmp = self._bbox_reg_head[lvl](hs[lvl])
            else:
                outputs_class = self._cls_head(hs[lvl])
                tmp = self._bbox_reg_head(hs[lvl])

            if reference.shape[-1] == 6:
                tmp += reference
            else:
                assert reference.shape[-1] == 3
                tmp[..., :3] += reference

            outputs_coord = tmp.sigmoid()
            # print(f"Output coord: {outputs_coord.shape}")
            # print(f"Output class: {outputs_class.shape}")
            # print(f"Output class: {outputs_class}")
            # print(f"Output coord: {outputs_coord}")
            # quit()

            if self.hybrid and self.training:
                outputs_classes.append(outputs_class[:, 0 : self.num_queries_one2one])
                outputs_classes_one2many.append(outputs_class[:, self.num_queries_one2one :])
                outputs_coords.append(outputs_coord[:, 0 : self.num_queries_one2one])
                outputs_coords_one2many.append(outputs_coord[:, self.num_queries_one2one :])
            else:
                outputs_classes.append(outputs_class)
                outputs_coords.append(outputs_coord)

        if self.hybrid and self.training:
            outputs_classes_one2many = torch.stack(outputs_classes_one2many)
            # tensor shape: [num_decoder_layers, bs, num_queries_one2many, num_classes]
            outputs_coords_one2many = torch.stack(outputs_coords_one2many)
            # tensor shape: [num_decoder_layers, bs, num_queries_one2many, 6]

        pred_logits = torch.stack(outputs_classes) # (bs, num_queries+num_noised_gt+num_dn, 6)
        pred_boxes = torch.stack(outputs_coords)

        if self.extra_classes > 0: # inc background class
            pred_logits = pred_logits[:, :, :, :self.num_classes_orig_dataset + 1]

        # dn post process
        if self.dn['dn_number'] > 0 and dn_meta is not None:
            pred_logits, pred_boxes = dn_post_process(
                pred_logits,
                pred_boxes,
                dn_meta,
                self._aux_loss,
                self._set_aux_loss,
            )
            
        # Segmentation prediction
        if  self._msa_seg: 
            # Retrieve segmentaion maps
            msa_seg = []
            for key, value in out_backbone.items():
                obj_ = re.sub('[^a-zA-Z]+', '', key)            
                if (obj_ in ['S']) and self._msa_seg:
                    if int(key[-1]) < int(self._input_level[-1]):
                        continue
                    else:
                        if self.extra_classes > 0: # inc background class
                            value = value[:, :self.num_classes_orig_dataset + 1, :, :, :]
                        msa_seg.append(value)
            pred_seg = msa_seg

            neck_enc_seg = []
            tmp = 0
            for hwd in spatial_shapes:
                hwd_tmp = torch.prod(hwd)
                item = memory[:,tmp:tmp+hwd_tmp,:]
                item = item.transpose(2,1)
                b, c = item.size()[0:2]
                item = item.view(b, c, *hwd)
                item = self._seg_neck(item)
                if self.extra_classes > 0: # inc background class
                    item = item[:, :self.num_classes_orig_dataset + 1]
                tmp += hwd_tmp
                neck_enc_seg.append(item)


        elif self._seg_proxy:
            seg_src = out_backbone['P0'] if (self._seg_proxy and self._backbone_name not in ['msavit']) else None 
            pred_seg = self._seg_head(seg_src) if (seg_src is not None) else []
        else:
            pred_seg = []

        
        if self.hybrid and self.training:
            out = {
                    'pred_logits': outputs_classes[-1],
                    'pred_boxes': outputs_coords[-1],
                    'pred_logits_one2many': outputs_classes_one2many[-1],
                    'pred_boxes_one2many': outputs_coords_one2many[-1],
                    'pred_seg': pred_seg
                    }
            if self._aux_loss:
                out["aux_outputs"] = self._set_aux_loss(outputs_classes, outputs_coords)
                out["aux_outputs_one2many"] = self._set_aux_loss(outputs_classes_one2many, outputs_coords_one2many)

        else: #################
            out = {
                'pred_logits': pred_logits[-1][:, : self.num_queries], # Take output of last layer
                'pred_boxes': pred_boxes[-1][:, : self.num_queries],
                'pred_seg': pred_seg,
                }

            if self._aux_loss:
                out['aux_outputs'] = self._set_aux_loss(pred_logits[:, : self.num_queries], pred_boxes[:, : self.num_queries])

        if  self._msa_seg: 
            out.update({'neck_enc_seg': neck_enc_seg})


        if self.two_stage:
            enc_outputs_coord = enc_outputs_coord_unact.sigmoid()
            out['enc_outputs'] = {'pred_logits': enc_outputs_class, 'pred_boxes': enc_outputs_coord}
        
        # calculate contrastive loss
        con_losses = {}
        if self.is_contrastive and num_epoch%2 == 0:
            if targets is not None:
                per_gt_num = [target["boxes"].shape[0] for target in targets]
                max_gt = max(per_gt_num)
                num_gts = sum(per_gt_num)
                if num_gts > 0:
                    for li in range(hs.shape[0]):
                        contrastive_loss = 0.0
                        projs = torch.cat((pred_logits[li], pred_boxes[li]), dim=-1) # (bs, num_queries+num_noised_gt, #classes+7)
                        gt_projs = self.projector(projs[:, self.num_queries :].detach()) # (bs, num_noised_gt, 256)
                        pred_projs = self.predictor(self.projector(projs[:, : self.num_queries])) # (bs, num_queries, 256)

                        outputs = {}
                        #noised_labels = pred_logits[li][:,self.num_queries :]
                        #noised_boxes = pred_boxes[li][:,self.num_queries :]
                        pred_labels_layer = pred_logits[li][:, : self.num_queries]
                        pred_boxes_layer = pred_boxes[li][:, : self.num_queries]
                        outputs['pred_logits'] = pred_labels_layer
                        outputs['pred_boxes'] = pred_boxes_layer

                        matched_indices = self.matcher(outputs, targets)
                        # num_gts x num_locs
                        pos_idxs = list(range(1, dn_meta["num_dn_group"] + 1))
                        for bi, idx in enumerate(matched_indices): # out["matched_indices"]
                            sim_matrix = (
                                self.similarity_f(
                                    gt_projs[bi].unsqueeze(1),
                                    pred_projs[bi].unsqueeze(0),
                                )
                                / self.tau
                            )
                            matched_pairs = torch.stack(idx, dim=-1)
                            neg_mask = projs.new_ones(self.num_queries).bool()
                            neg_mask[matched_pairs[:, 0]] = False
                            for pair in matched_pairs:
                                pos_mask = torch.tensor([int(pair[1] + max_gt * pi) for pi in pos_idxs], device=self.device)
                                pos_pair = sim_matrix[pos_mask, pair[0]].view(-1, 1)
                                neg_pairs = sim_matrix[:, neg_mask][pos_mask]
                                loss_gti = (
                                    torch.log(torch.exp(pos_pair) + torch.exp(neg_pairs).sum(dim=-1, keepdim=True))
                                    - pos_pair
                                )
                                contrastive_loss += loss_gti.mean()
                        con_losses[f"loss_contrastive_dec_{li}"] = self.contras_loss_coeff * contrastive_loss / num_gts
                        # print(con_losses[f"loss_contrastive_dec_{li}"])
        
        if self.training:
            return out, con_losses, dn_meta
        if self.hybrid: # reset num_queries for training
            self.num_queries = save_num_queries
        return out

    @torch.jit.unused
    def _set_aux_loss(self, pred_logits, pred_boxes):
        # Hack to support dictionary with non-homogeneous values
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(pred_logits[:-1], pred_boxes[:-1])]


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
