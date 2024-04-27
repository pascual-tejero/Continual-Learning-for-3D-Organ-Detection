"""Deformable DETR Transformer class adapted from https://github.com/fundamentalvision/Deformable-DETR."""

import copy
import math
import torch
import torch.nn.functional as F
from torch import nn

from transoar.models.ops.modules import MSDeformAttn





class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1,
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_cuda=True
    ):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, use_cuda)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention -> query and input_flatten are the same

        src2, _ = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src


class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):

            ref_z, ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, D_ - 0.5, D_, dtype=torch.float32, device=device),
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device)
            )
            # Get relative coords in range [0, 1], ref points in masked areas have values > 1
            ref_z = ref_z.reshape(-1)[None] / (valid_ratios[:, None, lvl, 2] * D_)
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y, ref_z), -1)    # Coords in format WHD/XYZ
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)  # [Batch, AllLvlPatches, RelativeRefCoords]
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # Valid ratio also in format WHD/XYZ
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src

        # Get reference points normalized and in valid areas, [Batch, AllLvlPatches, NumLevels, RefCoords]
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output





class DeformableTransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model=256,
        d_ffn=1024,
        dropout=0.1, 
        activation="relu",
        n_levels=4,
        n_heads=8,
        n_points=4,
        use_cuda=True,
        is_contrastive=False,
        num_classes=0,
    ):
        super().__init__()

        # cross attention
        self.nheads = n_heads
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, use_cuda)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.is_contrastive = is_contrastive
        
        # map a concatenated vector [reference, class] to positional embedding
        if self.is_contrastive:
            self.pos_embed_layer = MLP(num_classes+7, d_model, d_model, 3)  
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self, idx, tgt, query_pos, reference_points_def, src, src_spatial_shapes, level_start_index, src_padding_mask=None, reference_points=None, self_attn_mask=None):
        
        if self.is_contrastive:
            # reference_points_def is used in deformable attention
            # reference_points is used to generate query_pos
            if idx == 0:
                tgt = self.pos_embed_layer(reference_points)
                q = k = tgt

            elif query_pos is None:
                query_pos = self.pos_embed_layer(reference_points)
                q = k = self.with_pos_embed(tgt, query_pos)
        else:
            # self attention
            q = k = self.with_pos_embed(tgt, query_pos)
        
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1), \
                              attn_mask=self_attn_mask,)[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # cross attention
        tgt2, _ = self.cross_attn(self.with_pos_embed(tgt, query_pos),
                               reference_points_def,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DeformableTransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False, use_dab=False, 
                 d_model=256, high_dim_query_update=False, no_sine_embed=True, num_classes=0, dn=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        self.num_classes = num_classes
        # new
        self.bbox_embed = None
        self.class_embed = None
        self.use_dab = use_dab
        self.d_model = d_model
        self.no_sine_embed = no_sine_embed
        self.high_dim_query_update = high_dim_query_update
        self.dn = dn
        if dn:
            self.ref_point_head = MLP(2 * d_model, d_model, d_model, 2)
        if high_dim_query_update:
            self.high_dim_query_proj = MLP(d_model, d_model, d_model, 3)

    def forward(self, tgt, reference_points, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                query_pos=None, src_padding_mask=None, self_attn_mask=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        for lid, layer in enumerate(self.layers):
            if reference_points.shape[-1] == self.num_classes+7: # query contrast is enabled
                reference_points_def = reference_points[:, :, :6] # input for deformable attention
                reference_points_def = reference_points_def[:, :, None] * \
                torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            elif reference_points.shape[-1] == 6: # two_stage is enabled
                reference_points_def = reference_points[:, :, None] * \
                torch.cat([src_valid_ratios, src_valid_ratios], -1)[:, None]
            else: 
                assert reference_points.shape[-1] == 3
                reference_points_def = reference_points[:, :, None] * src_valid_ratios[:, None]
            
            if query_pos is None: # dn is enabled
                if reference_points.shape[-1] == 6:
                    query_sine_embed = gen_sineembed_for_position(reference_points_def[:, :, 0, :], self.d_model)
                    query_pos = self.ref_point_head(query_sine_embed) # bs, nq, d_model

            reference_points_input = reference_points_def
            output = layer(lid, output, query_pos, reference_points_input, src, src_spatial_shapes, src_level_start_index, src_padding_mask, reference_points, self_attn_mask)
            
            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)
                if reference_points.shape[-1] == self.num_classes+7: # query contrast is enabled
                    new_reference_points = tmp + inverse_sigmoid(reference_points[:, :, :6])
                    new_reference_points = new_reference_points.sigmoid()
                    new_class_scores = self.class_embed[lid](output).sigmoid()
                    reference_points = torch.cat(
                        (
                            new_reference_points.detach(),
                            new_class_scores.detach(),
                        ),
                        dim=-1,
                    )
                elif reference_points.shape[-1] == 6: # two_stage is enabled
                    new_reference_points = tmp + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()
                else:
                    assert reference_points.shape[-1] == 3
                    new_reference_points = tmp
                    new_reference_points[..., :3] = tmp[..., :3] + inverse_sigmoid(reference_points)
                    new_reference_points = new_reference_points.sigmoid()
                    reference_points = new_reference_points.detach()

            if self.return_intermediate:
                intermediate.append(output)
                if reference_points.shape[-1] == self.num_classes+7:
                    intermediate_reference_points.append(reference_points[:, :, :6])
                else:
                    assert reference_points.shape[-1] == 6 or reference_points.shape[-1] == 3
                    intermediate_reference_points.append(reference_points)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

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

def gen_sineembed_for_position(pos_tensor, d_model):
    assert d_model % 3 == 0
    num_pos_feats = d_model//3
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / num_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    z_embed = pos_tensor[:, :, 2] * scale  # Added z-coordinate embedding
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_z = z_embed[:, :, None] / dim_t  # Added z-coordinate position calculation
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_z = torch.stack((pos_z[:, :, 0::2].sin(), pos_z[:, :, 1::2].cos()), dim=3).flatten(2)  # Added z-coordinate position stacking

    if pos_tensor.size(-1) == 6:
        w_embed = pos_tensor[:, :, 3] * scale
        h_embed = pos_tensor[:, :, 4] * scale
        d_embed = pos_tensor[:, :, 5] * scale  # Added depth embedding
        pos_w = w_embed[:, :, None] / dim_t
        pos_h = h_embed[:, :, None] / dim_t
        pos_d = d_embed[:, :, None] / dim_t  # Added depth position calculation
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_d = torch.stack((pos_d[:, :, 0::2].sin(), pos_d[:, :, 1::2].cos()), dim=3).flatten(2)  # Added depth position stacking

        pos = torch.cat((pos_x, pos_y, pos_z, pos_w, pos_h, pos_d), dim=2)  # Updated concatenation for 6 coordinates
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))

    return pos

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)











class DeformableTransformer(nn.Module):
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_encoder_layers=6,
        num_decoder_layers=6, 
        dim_feedforward=1024,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False,
        num_feature_levels=4,
        dec_n_points=4,
        enc_n_points=4,
        use_cuda=True,
        use_encoder=True,
        use_dab=False,
        two_stage=False,
        two_stage_num_proposals=0,
        mom=0.999,
        num_classes=0,
        is_contrastive=False,
        dn=False,
    ):
        super().__init__()
        assert (not two_stage or not use_dab) or use_encoder, "use_encoder must be True when two_stage or use_dab is True" 
        self.d_model = d_model
        self.nhead = nhead
        self.use_encoder = use_encoder
        self.use_dab = use_dab
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals
        self.m = mom
        self.is_contrastive = is_contrastive
        self.dn = dn
        self.num_classes = num_classes 

        encoder_layer = DeformableTransformerEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, enc_n_points, use_cuda
        )
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DeformableTransformerDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels,
            nhead, dec_n_points, use_cuda, is_contrastive, num_classes
        )
        self.decoder = DeformableTransformerDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec,
                                                    use_dab=use_dab, d_model=d_model, high_dim_query_update=False,
                                                    no_sine_embed=False, num_classes=num_classes, dn=self.dn)
        if self.is_contrastive:
            self.decoder_gt = copy.deepcopy(self.decoder) # decoder to generate gt embeddings
        
        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if self.two_stage:
            self.enc_output = nn.Linear(self.d_model, self.d_model)
            self.enc_output_norm = nn.LayerNorm(self.d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
            
        if not self.is_contrastive and self.dn:
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        if self.dn: # for contrastive and dn
            self.reference_points = nn.Linear(d_model, 6) 
        else:
            self.reference_points = nn.Linear(d_model, 3)
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage and not self.use_dab:
            nn.init.xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            nn.init.constant_(self.reference_points.bias.data, 0.)
        nn.init.normal_(self.level_embed)
        
        if self.is_contrastive:
            for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False
        
    @torch.no_grad()
    def _momentum_update_gt_decoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.decoder.parameters(), self.decoder_gt.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    def get_valid_ratio(self, mask):
        _, D, H, W = mask.shape
        valid_D = torch.sum(~mask[:, :, 0, 0], 1)
        valid_H = torch.sum(~mask[:, 0, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, 0, :], 1)
        valid_ratio_d = valid_D.float() / D
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h, valid_ratio_d], -1)
        return valid_ratio
    
    def get_proposal_pos_embed(self, proposals):
        assert self.d_model % 3 == 0
        num_pos_feats = self.d_model//3
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 6
        proposals = proposals.sigmoid() * scale
        # N, L, 6, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 6(bbox), 64, 2(sine,cosine)
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos
    
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (D_, H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur : (_cur + D_ * H_ * W_)].view(
                N_, D_, H_, W_, 1
            )
            valid_D = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_H = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, 0, :], 1)

            grid_z, grid_y, grid_x = torch.meshgrid(
                torch.linspace(0, D_ - 1, D_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
            )
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1), grid_z.unsqueeze(-1)], -1)
            
            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1), valid_D.unsqueeze(-1)], 1).view(
                N_, 1, 1, 1, 3
            )
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1, -1) + 0.5) / scale
            whd = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)

            proposal = torch.cat((grid, whd), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += D_ * H_ * W_
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = (
            (output_proposals > 0.01) & (output_proposals < 0.99)
        ).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(
            memory_padding_mask.unsqueeze(-1), float("inf")
        )
        output_proposals = output_proposals.masked_fill(
            ~output_proposals_valid, float("inf")
        )

        output_memory = memory
        output_memory = output_memory.masked_fill(
            memory_padding_mask.unsqueeze(-1), float(0)
        )
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals
    
    def forward(self, srcs, masks, query_embed, pos_embeds, dn_mask=None, noised_gt_box=None, noised_gt_onehot=None, targets=None):
        assert self.two_stage or query_embed is not None
        
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate( zip(srcs, masks, pos_embeds) ):
            bs, c, d, h, w = src.shape
            spatial_shape = (d, h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)                                # [Batch, Patches, HiddenDim]   
            mask = mask.flatten(1)                                              # [Batch, Patches ]
            pos_embed = pos_embed.flatten(2).transpose(1, 2)                    # [Batch, Patches, HiddenDim]
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)    # [Batch, Patches, HiddenDim]
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)                                 # [Batch, AllLvlPatches, HiddenDim]
        mask_flatten = torch.cat(mask_flatten, 1)                               # [Batch, AllLvlPatches]
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)             # [Batch, AllLvlPatches, HiddenDim]

        # Shapes of feature maps of levels in use
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)

        # Determine indices of batches that mark the start of a new feature level
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        # Determine the ratios of valid regions based on the mask for in format WHD/XYZ
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        if self.use_encoder:
            memory = self.encoder(
                src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten, mask_flatten
            ) 
        else:
            memory = src_flatten   


        # prepare input for decoder
        bs, _, c = memory.shape                                                 # [Batch, AllLvlPatches, HiddenDim]
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)
            # output_memory: bs, num_tokens, c
            # output_proposals: bs, num_tokens, 6. unsigmoided.
            # output_proposals: bs, num_tokens, 6

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals
            topk = self.two_stage_num_proposals
            tok_scores, topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)
            tok_scores = tok_scores.unsqueeze(-1)

            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points
            if self.is_contrastive:
                query_embed, tgt = None, None # follow ConQueR, generate later in decoder layer
            else:
                pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact)))
                query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        elif self.use_dab:
            raise NotImplementedError('not implemeted dab-detr with query contrast yet')
        else:
            query_embed, tgt = torch.split(query_embed, c, dim=1)   # Tgt in contrast to detr not zeros, but learnable
            query_embed = query_embed.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1)
            reference_points = self.reference_points(query_embed).sigmoid()
            init_reference_out = reference_points
            if self.is_contrastive:
                # since two-stage is disabled, initialize class scores with random numbers
                tok_scores = torch.randn(reference_points.size(0), reference_points.size(1)).unsqueeze(-1)
                tok_scores = tok_scores.to(reference_points.device)
                query_embed, tgt = None, None # follow ConQueR, generate later in decoder layer
        
        if self.is_contrastive:
            reference_points = torch.cat((reference_points, tok_scores.detach().expand(-1, -1, self.num_classes+1)), dim=-1)
        
            if noised_gt_box is not None:
                noised_gt_proposals = torch.cat(
                    (
                        noised_gt_box, # (bs, num_noise_gt, 6)
                        noised_gt_onehot, # (bs, num_noise_gt, #num_classes+1)
                    ),
                    dim=-1,
                )

                reference_points = torch.cat(
                    (
                        noised_gt_proposals, # (bs, num_noise_gt, #num_classes+1+6)
                        reference_points, # (bs, num_queries, #num_classes+1+6)
                    ),
                    dim=1,
                )
            
            init_reference_out = reference_points[..., :6] 
        elif self.dn and noised_gt_onehot is not None:
            tgt = torch.cat([noised_gt_onehot, tgt], dim=1)
            reference_points = torch.cat([noised_gt_box, reference_points], dim=1)
            init_reference_out = reference_points
            query_embed = None # will generate in decoder
            # pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(reference_points)))
            # query_embed, _ = torch.split(pos_trans_out, c, dim=2)




        # decoder
        hs, inter_references_out = self.decoder(
            tgt, reference_points, memory, spatial_shapes, level_start_index, valid_ratios, 
            query_pos=query_embed if not self.use_dab else None, src_padding_mask=mask_flatten, self_attn_mask=dn_mask
        )


        
        if self.is_contrastive:
            # optional gt forward
            if targets is not None:
                batch_size = len(targets)
                per_gt_num = [target["boxes"].shape[0] for target in targets]
                max_gt_num = max(per_gt_num)

                batched_gt_boxes_with_score = memory.new_zeros(batch_size, max_gt_num, self.num_classes+7)
                for bi in range(batch_size):
                    batched_gt_boxes_with_score[bi, : per_gt_num[bi], :6] = targets[bi]["boxes"]
                    batched_gt_boxes_with_score[bi, : per_gt_num[bi], 6:] = F.one_hot(
                        targets[bi]["labels"], num_classes=self.num_classes+1
                    )

                with torch.no_grad():
                    self._momentum_update_gt_decoder()
                    if noised_gt_box is not None:
                        dn_group_num = noised_gt_proposals.shape[1] // (max_gt_num * 2)

                        pos_idxs = list(range(0, dn_group_num * 2, 2))
                        pos_noised_gt_proposals = torch.cat(
                            [noised_gt_proposals[:, pi * max_gt_num : (pi + 1) * max_gt_num] for pi in pos_idxs],
                            dim=1,
                        )
                        gt_proposals = torch.cat((batched_gt_boxes_with_score, pos_noised_gt_proposals), dim=1)

                        # create attn_mask for gt groups
                        gt_attn_mask = memory.new_ones(
                            (dn_group_num + 1) * max_gt_num, (dn_group_num + 1) * max_gt_num
                        ).bool()
                        for di in range(dn_group_num + 1):
                            gt_attn_mask[
                                di * max_gt_num : (di + 1) * max_gt_num,
                                di * max_gt_num : (di + 1) * max_gt_num,
                            ] = False
                    else:
                        gt_proposals = batched_gt_boxes_with_score
                        gt_attn_mask = None

                    hs_gt, inter_references_gt = self.decoder_gt(
                        None, # tgt
                        gt_proposals,
                        memory,
                        spatial_shapes,
                        level_start_index,
                        valid_ratios,
                        None, # query_pos
                        mask_flatten,
                        gt_attn_mask,
                    )

                init_reference_out = torch.cat(
                    (
                        init_reference_out,
                        gt_proposals[..., :6],
                    ),
                    dim=1,
                )

                hs = torch.cat(
                    (
                        hs,
                        hs_gt,
                    ),
                    dim=2,
                )

                inter_references_out = torch.cat(
                    (
                        inter_references_out,
                        inter_references_gt,
                    ),
                    dim=2,
                )



        datum = {
            'hs': hs, 
            'init_reference_out': init_reference_out, 
            'inter_references_out': inter_references_out,
            'spatial_shapes': spatial_shapes, 
            'memory': memory
        }


        if self.two_stage:
            datum.update({
                'enc_outputs_class': enc_outputs_class,
                'enc_outputs_coord_unact': enc_outputs_coord_unact,
                }) 
        
        return datum
