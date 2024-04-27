# Continual Learning for 3D Organ Detection
Master thesis project at the Technical University of Munich (TUM). Still in progress...


# New config parameters
DAB & two-stage (configure "neck" sub-configuration)
- use_dab: True 
- num_patterns: 3 # number of patterns for anchors, DAB-DETR uses 3
- two_stage: True
- box_refine: True

Custom matching (place these below 'set_cost_giou')
- dense_matching: True
- dense_matching_lambda: 0.5 

Hybrid matching (from [H-Deformable-DETR](https://github.com/HDETR/H-Deformable-DETR), place these below 'set_cost_giou')
- hybrid_matching: True
- hybrid_K: 6   # repeated GT
- hybrid_T: 300 # additional many2one queries
- hybrid_loss_weight_one2many: 1

Hybrid matching with DM in additional branch
- hybrid_matching: True
- hybrid_dense_matching: True # also set hybrid_matching to True, disregards hybrid_K
- hybrid_dense_matching_lambda: 0.5
- hybrid_T: 300 # additional many2one queries
- hybrid_loss_weight_one2many: 1

Class-specific queries (place these below 'set_cost_giou')
- class_matching: True
- class_matching_query_split: [10,20,20,30,20] these are the number of queries for class 1,2,3,... They need to sum up to num_queries.

Focal loss for class loss (place before 'loss_coefs')
- focal_loss: True

Contrastive learning & denoising for queries (place as sub-config inside neck)
- contrastive:
    - enabled: True
    - mom: 0.999
    - dim: 256
    - eqco: 1000
    - tau: 0.7
    - loss_coeff: 0.2
- dn:
    - enabled: True
    - dn_number: 3  # number of dn groups 
    - dn_box_noise_ratio: 0.2 # 0.4 
    - dn_label_noise_ratio: 0.25 # 0.5

# Visualization
Arguments for test script:
- --run NAME_OF_RUN
- --num_gpu CUDA_ID → e.g., 0
- --val → use validation dataset
- --last → use last model instead of best
- --per_sample_results → exports json with results for every individual case
- --save_preds → save visualizations
    - --vis_mode o3d (default) OR nii 
        - o3d: exports ply files, can be visualized by `display_case_ply.py`
        - nii: exports nii.gz file for segmentation map and markup jsons for boxes, can be visualized by 3D Slicer
    - --exp_img → additionally exports input image as nii.gz, only works with `--vis_mode nii`
    - --save_attn_map → additionally exports sampling locations and reference points of last cross-attention layer in decoder

Examples:

- export ply: `python scripts/test.py --run myrun --num_gpu 0 --save_preds`
- export nifti: `python scripts/test.py --run myrun --num_gpu 0 --save_preds --vis_mode nii` 
- export ply + sampling loc: `python scripts/test.py --run myrun --num_gpu 0 --save_attn_map`
- export nifti + sampling loc: `python scripts/test.py --run myrun --num_gpu 0 --save_attn_map --vis_mode nii` 
- export nifti + input image: `python scripts/test.py --run myrun --num_gpu 0 --save_preds --vis_mode nii --exp_img` 

# Loading nifti exports in 3D Slicer
Tested with 3D Slicer 5.2.2
1. Download & open [3D Slicer](https://download.slicer.org/)
2. Click on "Add Data" / Ctrl+O
3. Choose case directory and put "Segmentation" for "seg_mask.nii.gz" → OK
4. Run this code in the python console (Ctrl+3) to deactivate markup fillings and configure white background: 

    ```
    viewNode = slicer.app.layoutManager().threeDWidget(0).mrmlViewNode() 
    viewNode.SetBackgroundColor(1,1,1) 
    viewNode.SetBackgroundColor2(1,1,1) 
    for node in slicer.mrmlScene.GetNodesByClass('vtkMRMLMarkupsROINode'): 
        displayNode = node.GetDisplayNode() 
        if displayNode: 
            displayNode.SetFillVisibility(False)
    ```
5. Go to the "Segmentations" module and click on "Show 3D" for seg_mask.nii.gz
6. Orientation marker and 3D cube can be disabled in the 3D view window itself (pin on top-left). Since the training data does not contain metadata, the orientation markers and voxel spacing might be wrong.
