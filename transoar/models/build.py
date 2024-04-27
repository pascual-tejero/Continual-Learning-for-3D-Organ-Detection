"""Module containing functionality to build different parts of the model."""

from transoar.models.matcher import HungarianMatcher
from transoar.models.criterion import TransoarCriterion
from transoar.models.backbones.attn_fpn.attn_fpn import AttnFPN
from transoar.models.necks.def_detr_transformer import DeformableTransformer
from transoar.models.position_encoding import PositionEmbeddingSine3D, PositionEmbeddingLearned3D

from transoar.models.backbones.resnet3d import ResNet3D
from transoar.models.backbones.MSAViT import MSAViT
from transoar.models.backbones.swin_unetr import Swin_UNETR

from transoar.utils.io import load_json
from pathlib import Path
import os

def build_backbone(config):
    if config['name'].lower() in ['attn_fpn']:
        return AttnFPN(config)
    elif config['name'].lower() in ['msavit']:
        return MSAViT(config)
    elif config['name'].lower() in ['resnet']:
        return ResNet3D(config)
    elif config['name'].lower() in ['swin_unetr']:
        return Swin_UNETR(config)

def build_neck(config):
    model = DeformableTransformer(
        d_model=config['hidden_dim'],
        nhead=config['nheads'],
        num_encoder_layers=config['enc_layers'],
        num_decoder_layers=config['dec_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        activation="relu",
        return_intermediate_dec=True,
        dec_n_points=config['dec_n_points'],
        enc_n_points=config['enc_n_points'],
        use_cuda=config['use_cuda'],
        use_encoder=config['use_encoder'],
        num_feature_levels=config['num_feature_levels'],
        use_dab=config.get('use_dab', False),
        two_stage=config.get('two_stage', False),
        two_stage_num_proposals=config['num_queries'],
        num_classes=config['num_classes'],
        is_contrastive=config.get('contrastive', {}).get('enabled', False),
        mom=config['contrastive']['mom'] if config.get('contrastive', {}).get('enabled', False) else None,
        dn=config.get('dn', {}).get('enabled', False),
    ) 

    return model

def build_criterion(config):
    qs = config.get('class_matching_query_split', [])
    if qs != [] and config.get('class_matching', False):
        assert sum(qs) == config['neck']['num_queries'], "query split doesn't match num_queries"

    if config.get('hybrid_dense_matching', False):
        hybrid_dense_matcher = HungarianMatcher(
            cost_class=config['set_cost_class'],
            cost_bbox=config['set_cost_bbox'],
            cost_giou=config['set_cost_giou'],
            dense_matching=True,
            dense_matching_lambda=config.get('hybrid_dense_matching_lambda', 0.5),
            class_matching=config.get('hybrid_dense_class_matching', False),
            class_matching_query_split=config.get('hybrid_dense_class_matching_query_split', []),
            recursive_dm_dn=config['neck'].get('dn', {}).get('enabled', False) # if dn and dm are enabled, use them recursively
        )

        hybrid_dense_criterion = TransoarCriterion(
            num_classes=config['neck']['num_classes'],
            matcher=hybrid_dense_matcher,
            seg_proxy=config['backbone']['use_seg_proxy_loss'] and not config['backbone'].get('use_msa_seg_loss', False),
            seg_fg_bg=config['backbone']['fg_bg'],
            seg_msa=config['backbone'].get('use_msa_seg_loss', False),
            focal_loss=config.get('focal_loss', False),
            config=config
        )

    # Check if there is extra classes in the dataset
    data_path = os.environ.get('TRANSOAR_DATA')
    data_dir = Path(data_path).resolve()
    data_config = load_json(data_dir / config['dataset'] / "data_info.json")
    num_classes = len(data_config['labels'])
    extra_classes = config["backbone"]["num_organs"] - num_classes
    num_classes_orig_dataset = len(data_config['labels'])
   
    matcher = HungarianMatcher(
        cost_class=config['set_cost_class'],
        cost_bbox=config['set_cost_bbox'],
        cost_giou=config['set_cost_giou'],
        dense_matching=config.get('dense_matching', False),
        dense_matching_lambda=config.get('dense_matching_lambda', 0.5),
        class_matching=config.get('class_matching', False),
        class_matching_query_split=config.get('class_matching_query_split', []),
        recursive_dm_dn=config['neck'].get('dn', {}).get('enabled', False), # if dn and dm are enabled, use them recursively
        extra_classes=extra_classes,
        num_classes_orig_dataset=num_classes_orig_dataset,
        config=config
    )

    criterion = TransoarCriterion(
        num_classes=config['neck']['num_classes'],
        matcher=matcher,
        seg_proxy=config['backbone']['use_seg_proxy_loss'] and not config['backbone'].get('use_msa_seg_loss', False),
        seg_fg_bg=config['backbone']['fg_bg'],
        seg_msa=config['backbone'].get('use_msa_seg_loss', False),
        focal_loss=config.get('focal_loss', False),
        extra_classes=extra_classes,
        num_classes_orig_dataset=num_classes_orig_dataset, 
        config=config
    )

    if config.get('hybrid_dense_matching', False):
        return criterion, hybrid_dense_criterion
    else:
        return criterion

def build_pos_enc(config):
    channels = config['hidden_dim']
    if config['pos_encoding'] == 'sine':
        return PositionEmbeddingSine3D(channels=channels)
    elif config['pos_encoding'] == 'learned':
        return PositionEmbeddingLearned3D(channels=channels)
    else:
        raise ValueError('Please select a implemented pos. encoding.')
