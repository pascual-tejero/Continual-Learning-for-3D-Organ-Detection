"""Script to evalute performance on the val and test set."""

import os,sys
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import json
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="TypedStorage")

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
print("append to path & chdir:", base_dir)
os.chdir(base_dir)
sys.path.append(base_dir)


from transoar.utils.io import load_json, write_json
try:
    from transoar.utils.visualization import save_pred_visualization, save_pred_visualization_nifti
    import open3d as o3d
    import matplotlib.pyplot as plt
except:
    pass
from transoar.data.dataloader import get_loader
from transoar.models.transoarnet import TransoarNet
from transoar.evaluator import DetectionEvaluator, SegmentationEvaluator
from transoar.inference import inference
from transoar.utils.bboxes import box_cxcyczwhd_to_xyzxyz, iou_3d
from scripts.train import match

class Tester:

    def __init__(self, args):
        path_to_run = Path('./runs/' + args.run)
        self.config = load_json(path_to_run / 'config.json')

        self._save_preds = args.save_preds
        self._save_attn_map = args.save_attn_map
        self._save_msa_attn_map = args.save_msa_attn_map
        self._per_sample_results = args.per_sample_results
        self._vis_mode = args.vis_mode
        self._exp_img = args.exp_img
        self._class_dict = self.config['labels']
        self._segm_eval = self.config['backbone']['use_seg_proxy_loss']

        if self._save_attn_map and not self._save_preds:
            self._save_preds = True
            print('Activated exporting predictions because export sampling locations was configured.')
        
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.num_gpu)
        self._device = 'cuda' if args.num_gpu >= 0 else 'cpu'

        # Get path to checkpoint
        avail_checkpoints = [path for path in path_to_run.iterdir() if 'model_' in str(path)]
        avail_checkpoints.sort(key=lambda x: len(str(x)))

        for a in avail_checkpoints:
            if args.model_load == str(a):
                path_to_ckpt = a
                break
        else:
            if args.model_load == 'last':
                path_to_ckpt = [path for path in avail_checkpoints if 'last' in str(path)]
            elif args.model_load == 'best':
                path_to_ckpt = [path for path in avail_checkpoints if 'best' in str(path)]
            elif args.model_load == 'best_val':
                path_to_ckpt = [path for path in avail_checkpoints if 'best_val' in str(path)]
            elif args.model_load == 'best_test':
                path_to_ckpt = [path for path in avail_checkpoints if 'best_test' in str(path)]
            elif isinstance(int(args.model_load), int):
                path_to_ckpt = [path for path in avail_checkpoints if args.model_load in str(path)]

        
            if len(path_to_ckpt) == 0:
                raise ValueError('No checkpoint found for specified epoch.')
            path_to_ckpt = path_to_ckpt[0]

        print(f'Loading checkpoint: {path_to_ckpt}')

        # Build necessary components
        self._set_to_eval = 'val' if args.val else 'test'
        self._test_loader = get_loader(self.config, self._set_to_eval, batch_size=1, test_script=True)

        self._evaluator = DetectionEvaluator(
            classes=list(self.config['labels'].values()),
            classes_small=self.config['labels_small'],
            classes_mid=self.config['labels_mid'],
            classes_large=self.config['labels_large'],
            iou_range_nndet=(0.1, 0.5, 0.05),
            iou_range_coco=(0.5, 0.95, 0.05),
            sparse_results=False
        )

        self._segm_evaluator = SegmentationEvaluator(seg_fg_bg=self.config['backbone']['fg_bg'],
                                                     ce_dice=self._segm_eval, 
                                                     hd95=self._segm_eval)

        self._model = TransoarNet(self.config).to(device=self._device)

        # Load checkpoint
        checkpoint = torch.load(path_to_ckpt, map_location=self._device)
        if self.config['backbone']['name'].lower() == "resnet": # fix renamed projection layers for runs pre-e23ce8b4
            fixed_dict = {}
            for k, v in checkpoint['model_state_dict'].items():
                if "_512to258" in k: 
                    fixed_dict[k.replace('_backbone._512to258','_backbone._out1')] = v
                elif "_1024to258" in k: 
                    fixed_dict[k.replace('_backbone._1024to258','_backbone._out2')] = v
                elif "_2048to258" in k: 
                    fixed_dict[k.replace('_backbone._2048to258','_backbone._out3')] = v
                else:
                    fixed_dict[k] = v
            checkpoint['model_state_dict'] = fixed_dict

        try:
            self._model.load_state_dict(checkpoint['model_state_dict'],strict=True)
        except Exception as e:
            print("These layers could not be loaded: ", e)
            print("Loading pretrained model with strict=False ...")
            self._model.load_state_dict(checkpoint['model_state_dict'],strict=False)
        
        self._model.eval()

        # Create dir to store results
        self._path_to_results = path_to_run / 'results' / path_to_ckpt.parts[-1][:-3]
        self._path_to_results.mkdir(parents=True, exist_ok=True)

        if self._save_preds:
            self._path_to_vis = self._path_to_results / ('vis_' + self._set_to_eval)
            self._path_to_vis.mkdir(parents=False, exist_ok=True)
  
    def run(self):
        if self._save_attn_map:
            backbone_attn_weights_list = []
            reference_points_list, sampling_points_list, dec_attn_weights_list = [], [], []
            
            # Register hooks to efficiently access relevant weights
            hooks = [
                self._model._neck.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: (reference_points_list.append(output[1][2].detach().clone()))
                ),
                self._model._neck.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: (sampling_points_list.append(output[1][1].detach().clone()))
                ),
                self._model._neck.decoder.layers[-1].cross_attn.register_forward_hook(
                    lambda self, input, output: (dec_attn_weights_list.append(output[1][0].detach().clone()))
                )
            ]
            if self._model._backbone_name.lower() == "msavit" and self._save_msa_attn_map:
                # For visualizing attention map in MSA backbone
                hooks.append(self._model._backbone._decoder.msa_dec[-1].levels[-1].blocks[-1].attn.register_forward_hook(
                        lambda self, input, output: (backbone_attn_weights_list.append(output[-1].detach().clone())))
                        )
        per_sample_results = {}
        with torch.no_grad():
            for idx, (data, mask, bboxes, seg_mask, paths) in enumerate(tqdm(self._test_loader)):
                # Put data to gpu
                #print("processing ", paths)
                data, mask = data.to(device=self._device), mask.to(device=self._device)
            
                targets = {
                    'boxes': bboxes[0][0].to(dtype=torch.float, device=self._device),
                    'labels': bboxes[0][1].to(device=self._device)
                }

                # Only use complete data for performance evaluation
                #if targets['labels'].shape[0] < len(self._class_dict):
                #    continue

                # Make prediction
                out = self._model(data)

                # Format out to fit evaluator and estimate best predictions per class
                if self._save_attn_map:
                    pred_boxes, pred_classes, pred_scores, query_ids = inference(out, vis_queries=self._save_attn_map)
                else:
                    pred_boxes, pred_classes, pred_scores = inference(out)
                    
                gt_boxes = [targets['boxes'].detach().cpu().numpy()]
                gt_classes = [targets['labels'].detach().cpu().numpy()]

                # Add pred to evaluator
                self._evaluator.add(
                    pred_boxes=pred_boxes,
                    pred_classes=pred_classes,
                    pred_scores=pred_scores,
                    gt_boxes=gt_boxes,
                    gt_classes=gt_classes
                )

                if self._per_sample_results:
                    sample_name = paths[0].stem + f'_case{idx}'
                    per_sample_results[sample_name] = {}
                    assert len(gt_classes) == 1 # batch size == 1
                    keys_as_int = [int(key) for key in self.config['labels'].keys()]
                    unique_classes = np.unique(keys_as_int)
                    for class_ in unique_classes:
                        pred_miss = False
                        gt_miss = False
                        pred_id = np.where(pred_classes[0]==class_)
                        if len(pred_id[0]) == 0: # no prediction
                            pred_miss = True
                        else:
                            pred_id = pred_id[0]
                        gt_id = np.where(gt_classes[0]==class_)
                        if len(gt_id[0]) == 0: # no gt
                            gt_miss = True
                        else:
                            gt_id = gt_id[0]
                        if gt_miss and pred_miss:
                            result = 'TN'
                        elif gt_miss:
                            result = 'FP'
                            print(f"no GT for {class_} in {sample_name}")
                        elif pred_miss:
                            result = 'FN'
                            print(f"no prediction for {class_} in {sample_name}")
                        else: # return IOU for TP
                            pred_box = torch.tensor(pred_boxes[0][pred_id])
                            gt_box = torch.tensor(gt_boxes[0][gt_id])
                            result,_ = iou_3d(box_cxcyczwhd_to_xyzxyz(pred_box), box_cxcyczwhd_to_xyzxyz(gt_box))
                            result = str(result.item())[:4] + ' IoU'
                        per_sample_results[sample_name][self.config['labels'][str(class_)]] = result

                if self._save_preds:
                    if self._vis_mode == 'o3d':
                        save_pred_visualization(
                            pred_boxes[0], pred_classes[0], gt_boxes[0], gt_classes[0], seg_mask[0], 
                            self._path_to_vis, self._class_dict, idx
                        )
                    elif self._vis_mode == 'nii':
                        input_image = None
                        if self._exp_img:
                            input_image = data
                        save_pred_visualization_nifti(
                            pred_boxes[0], pred_classes[0], gt_boxes[0], gt_classes[0], seg_mask[0], 
                            self._path_to_vis, self._class_dict, idx, input_image, self.config
                        )
                    else:
                        raise ValueError('Please select o3d or nii to export either ply or nii.gz files')

                if self._save_attn_map:
                    # Get returns from def attn hook
                    corner_points = np.array([[0, 0, 0], # corner points for plot
                                              [1, 0, 0],
                                              [0, 1, 0],
                                              [0, 0, 1],
                                              [1, 1, 0],
                                              [1, 0, 1],
                                              [0, 1, 1],
                                              [1, 1, 1]])
                    # Export MSA attn weights
                    if self._model._backbone_name.lower() == "msavit" and self._save_msa_attn_map:
                        attn_weights = backbone_attn_weights_list.pop(-1).squeeze().cpu().numpy()
                        np.save(f"{paths[0].stem}_attn_weights.npy",attn_weights)
                    
                    dec_attn_weights = dec_attn_weights_list.pop(-1).squeeze()
                    sampling_points = sampling_points_list.pop(-1).squeeze()
                    reference_points = reference_points_list.pop(-1).squeeze()
                    for key in query_ids:
                        
                        dec_attn_weights_np = dec_attn_weights.cpu().numpy()[query_ids[key],:,:,:]
                        
                        # batch, queries, attn_heads, feature_lvls, sampl_pt_per_query, 3D point
                        sampling_points_np = sampling_points.cpu().numpy()[query_ids[key],:,:,:,:]
                        reference_points_np = reference_points.cpu().numpy()[query_ids[key],:,:]

                        """
                        #array1_sum = np.sum(dec_attn_weights_np, axis=-1)
                        #array1_max = np.max(dec_attn_weights_np, axis=-1)
                        indices = np.argmax(array1_max, axis=-1)
                        index_array = np.indices(indices.shape)
                        index_array = np.concatenate([index_array, indices[np.newaxis, :]], axis=0)
                        sampling_points_np = sampling_points_np[tuple(index_array)]"""

                        dec_attn_weights_np_flat = dec_attn_weights_np.reshape(-1)
                        reshaped_points_s = sampling_points_np.reshape(-1, 3)
                        
                        # Create a mask of elements in array1_flat that are larger than x.x
                        mask = dec_attn_weights_np_flat > 0.1

                        # Create histogramm of attention weights
                        """plt.hist(dec_attn_weights_np_flat, bins=50)
                        plt.title('Distribution of Elements in Attention Weight Array')
                        plt.xlabel('Weights')
                        plt.ylabel('Frequency')
                        plt.savefig("attnweightdist.pdf")"""

                        # Apply the mask to array2_flat
                        reshaped_points_s = reshaped_points_s[mask]

                        # remove points outside of image
                        clip_mask =  (sampling_points_np >= 0).all(axis=-1) & (sampling_points_np <= 1).all(axis=-1)
                        sampling_points_np = sampling_points_np[clip_mask]

                        # Reshape the numpy array to have 3 columns (one for each dimension)
                        reshaped_points = np.reshape(sampling_points_np, (-1, 3))
                        reference_points_np = np.reshape(reference_points_np, (-1, 3))

                        # For undoing normalization
                        xyzxyz_scaling = np.array([seg_mask[0].shape[-3:]]).flatten()
                        #print(xyzxyz_scaling)
                        reshaped_points *= xyzxyz_scaling
                        reference_points_np *= xyzxyz_scaling
                        #print(np.max(reshaped_points[:,0]), np.min(reshaped_points[:,0]))
                        #print(np.max(reshaped_points[:,1]), np.min(reshaped_points[:,1]))
                        #print(np.max(reshaped_points[:,2]), np.min(reshaped_points[:,2]))

                        #print(reference_points_np)
                        
                        if self._vis_mode == 'o3d':
                            pcd = o3d.geometry.PointCloud()
                            pcd_ref = o3d.geometry.PointCloud()
                            pcd_corner = o3d.geometry.PointCloud()

                            pcd.points = o3d.utility.Vector3dVector(reshaped_points)
                            pcd_ref.points = o3d.utility.Vector3dVector(reference_points_np)
                            pcd_corner.points = o3d.utility.Vector3dVector(corner_points * xyzxyz_scaling)

                            pcd_corner.paint_uniform_color([1, 1, 0])   # yellow
                            pcd_ref.paint_uniform_color([0, 0, 0])      # black
                            pcd.paint_uniform_color([0, 1, 1])          # cyan
                            #cmap = plt.get_cmap('binary')
                            #colors = cmap(dec_attn_weights_np_flat)[:, :3] 
                            #pcd.colors = o3d.utility.Vector3dVector(colors)

                            plc_path = os.path.join(self._path_to_vis, 'case_' + str(idx))
                            o3d.io.write_point_cloud(os.path.join(plc_path ,f"{self._class_dict[str(key)]}-SL_.ply"), pcd)
                            o3d.io.write_point_cloud(os.path.join(plc_path ,f"{self._class_dict[str(key)]}-RP_.ply"), pcd_ref)

                        elif self._vis_mode == 'nii':
                            with open('config/template_sampling_locations.mrk.json', 'r') as f:
                                json_data_SL = json.load(f)
                            with open('config/template_reference_points.mrk.json', 'r') as f:
                                json_data_RP = json.load(f)
                            sampleLocations = []
                            referencePoints = []
                            for n in range(reshaped_points.shape[0]):
                                sampleLocations.append({
                                    "id": f"{n}",
                                    "label": f"SL-{n}",
                                    "description": "",
                                    "associatedNodeID": "",
                                    "position": [float(reshaped_points[n][0]), float(reshaped_points[n][1]), float(reshaped_points[n][2])],
                                    "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                    "selected": True,
                                    "locked": True,
                                    "visibility": True,
                                    "positionStatus": "defined"
                                })
                            for n in range(reference_points_np.shape[0]):
                                referencePoints.append({
                                    "id": f"{n}",
                                    "label": f"RP-{n}",
                                    "description": "",
                                    "associatedNodeID": "",
                                    "position": [float(reference_points_np[n][0]), float(reference_points_np[n][1]), float(reference_points_np[n][2])],
                                    "orientation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                                    "selected": True,
                                    "locked": True,
                                    "visibility": True,
                                    "positionStatus": "defined"
                                })
                            json_data_SL['markups'][0]['controlPoints'] = sampleLocations
                            json_data_RP['markups'][0]['controlPoints'] = referencePoints
                            plc_path = os.path.join(self._path_to_vis, 'case_' + str(idx))

                            # prepare size string for output files
                            if str(key) in self.config['labels_small']:
                                size = 's'
                            elif str(key) in self.config['labels_mid']:
                                size = 'm'
                            elif str(key) in self.config['labels_large']:
                                size = 'l'
                            
                            with open(os.path.join(plc_path ,f"{size}-{str(key).zfill(2)}-{self._class_dict[str(key)]}_SL.json"), 'w') as json_file:
                                json.dump(json_data_SL, json_file, indent=4)
                            with open(os.path.join(plc_path ,f"{size}-{str(key).zfill(2)}-{self._class_dict[str(key)]}_RP.json"), 'w') as json_file:
                                json.dump(json_data_RP, json_file, indent=4)
                    if self._vis_mode == 'o3d':    
                        o3d.io.write_point_cloud(os.path.join(plc_path ,"corner points_.ply"), pcd_corner)


            # Get and store final results
            # [torch.tensor([id_ for score, id_ in query_info[c]]).unique().shape for c in query_info.keys()]
            metric_scores = self._evaluator.eval()
            if self._segm_eval:
                metrics_segmentation = self._segm_evaluator.eval(out,seg_mask)
                metric_scores.update(metrics_segmentation)

            # Analysis of model parameter distribution
            num_params = sum(p.numel() for p in self._model.parameters() if p.requires_grad)
            num_backbone_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['backbone', 'input_proj', 'skip']))
            num_neck_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['neck', 'query']))
            num_head_params = sum(p.numel() for n, p in self._model.named_parameters() if p.requires_grad and match(n, ['head']))

            num_params_dict ={'num_params': num_params,
                      'num_backbone_params': num_backbone_params,
                      'num_neck_params': num_neck_params,
                      'num_head_params': num_head_params
                      }
            metric_scores.update(num_params_dict)  # Add parameters to result log

            write_json(metric_scores, self._path_to_results / ('results_' + self._set_to_eval + '.json'))
            if self._per_sample_results:
                write_json(per_sample_results, self._path_to_results / ('per_sample_results_' + self._set_to_eval + '.json'))
            


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add necessary args
    parser.add_argument('--run', required=True, type=str, help='Name of experiment in ./runs.')
    parser.add_argument('--num_gpu', type=int, default=-1, help='Use model_last instead of model_best.')
    parser.add_argument('--val', action='store_true', help='Evaluate performance on test set.')
    parser.add_argument('--model_load', type=str, default='last', help='Load model from checkpoint. Options: last, best_val, best_test, epoch number.')
    parser.add_argument('--save_preds', action='store_true', help='Save predictions.')
    parser.add_argument('--save_attn_map', action='store_true', help='Saves sampling locations of predictions.')
    parser.add_argument('--per_sample_results', action='store_true', help='Saves per sample results of predictions.')
    parser.add_argument('--vis_mode', type=str, default="o3d", help='Set type of visualization. \'o3d\' (default) or \'nii\' .')
    parser.add_argument('--exp_img', action='store_true', help='Exports input image as nii.gz. Only works with vis_mode==nii.')
    parser.add_argument('--save_msa_attn_map', action='store_true', help='Exports attn weights of msa backbone as npy.')
    args = parser.parse_args()

    tester = Tester(args)
    tester.run()
