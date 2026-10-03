import os
import tqdm

import torch
from mmcv.ops import box_iou_rotated
from lib.helpers.save_helper import load_checkpoint
from lib.helpers.decode_helper import extract_dets_from_outputs
from lib.helpers.decode_helper import decode_detections
import time


class Tester(object):
    def __init__(self, cfg, model, dataloader, logger, train_cfg=None, model_name='monomip'):
        self.cfg = cfg
        self.model = model
        self.dataloader = dataloader
        self.max_objs = dataloader.dataset.max_objs    # max objects per images, defined in dataset
        self.class_name = dataloader.dataset.class_name
        self.output_dir = os.path.join('./' + train_cfg['save_path'], model_name)
        self.dataset_type = cfg.get('type', 'KITTI')
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger
        self.train_cfg = train_cfg
        self.model_name = model_name

    def _should_evaluate(self):
        return getattr(self.dataloader.dataset, 'split', None) != 'test'

    def _bev_nms_preds(self, preds, iou_thresh):
        if iou_thresh is None or len(preds) <= 1:
            return preds

        kept = []
        for cls_id in sorted({int(pred[0]) for pred in preds}):
            cls_items = [pred for pred in preds if int(pred[0]) == cls_id]
            if len(cls_items) <= 1:
                kept.extend(cls_items)
                continue

            boxes = torch.tensor(
                [[float(pred[9]), float(pred[11]), float(pred[7]),
                  float(pred[8]), float(pred[12])]
                 for pred in cls_items],
                dtype=torch.float32,
            )
            scores = torch.tensor(
                [float(pred[-1]) for pred in cls_items],
                dtype=torch.float32,
            )
            ious = box_iou_rotated(boxes, boxes).cpu()
            order = torch.argsort(scores, descending=True).tolist()
            suppressed = torch.zeros(len(cls_items), dtype=torch.bool)

            for idx in order:
                if bool(suppressed[idx]):
                    continue
                kept.append(cls_items[idx])
                suppressed |= ious[idx] > float(iou_thresh)
                suppressed[idx] = False

        return sorted(kept, key=lambda pred: float(pred[-1]), reverse=True)

    def test(self):
        assert self.cfg['mode'] in ['single', 'all']

        # test a single checkpoint
        if self.cfg['mode'] == 'single' or not self.train_cfg["save_all"]:
            if self.train_cfg["save_all"]:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_epoch_{}.pth".format(self.cfg['checkpoint']))
            else:
                checkpoint_path = os.path.join(self.output_dir, "checkpoint_best.pth")
            assert os.path.exists(checkpoint_path)
            load_checkpoint(model=self.model,
                            optimizer=None,
                            filename=checkpoint_path,
                            map_location=self.device,
                            logger=self.logger)
            self.model.to(self.device)
            self.inference()
            if self._should_evaluate():
                self.evaluate()
            else:
                self.logger.info("==> Skipping local evaluation for KITTI test split; submission files are saved.")

        # test all checkpoints in the given dir
        elif self.cfg['mode'] == 'all' and self.train_cfg["save_all"]:
            start_epoch = int(self.cfg['checkpoint'])
            checkpoints_list = []
            for _, _, files in os.walk(self.output_dir):
                for f in files:
                    if f.endswith(".pth") and int(f[17:-4]) >= start_epoch:
                        checkpoints_list.append(os.path.join(self.output_dir, f))
            checkpoints_list.sort(key=os.path.getmtime)

            for checkpoint in checkpoints_list:
                load_checkpoint(model=self.model,
                                optimizer=None,
                                filename=checkpoint,
                                map_location=self.device,
                                logger=self.logger)
                self.model.to(self.device)
                self.inference()
                if self._should_evaluate():
                    self.evaluate()
                else:
                    self.logger.info("==> Skipping local evaluation for KITTI test split; submission files are saved.")

    def inference(self):
        torch.set_grad_enabled(False)
        self.model.eval()

        results = {}
        progress_bar = tqdm.tqdm(total=len(self.dataloader), leave=True, desc='Evaluation Progress')
        model_infer_time = 0
        for batch_idx, (inputs, calibs, targets, info) in enumerate(self.dataloader):
            # load evaluation data and move data to GPU.
            inputs = inputs.to(self.device)
            calibs = calibs.to(self.device)
            img_sizes = info['img_size'].to(self.device)
            if isinstance(targets, dict):
                for key in targets.keys():
                    targets[key] = targets[key].to(self.device)
            start_time = time.time()
            ###dn
            outputs = self.model(inputs, calibs, targets, img_sizes, dn_args = 0)
            ###
            end_time = time.time()
            model_infer_time += end_time - start_time

            dets = extract_dets_from_outputs(outputs=outputs, K=self.max_objs, topk=self.cfg['topk'])

            dets = dets.detach().cpu().numpy()

            # get corresponding calibs & transform tensor to numpy
            calibs = [self.dataloader.dataset.get_calib(index) for index in info['img_id']]
            info = {key: val.detach().cpu().numpy() for key, val in info.items()}
            cls_mean_size = self.dataloader.dataset.cls_mean_size
            dets = decode_detections(
                dets=dets,
                info=info,
                calibs=calibs,
                cls_mean_size=cls_mean_size,
                threshold=self.cfg.get('threshold', 0.2))

            bev_nms_iou = self.cfg.get('bev_nms_iou', None)
            if bev_nms_iou is not None:
                dets = {
                    img_id: self._bev_nms_preds(preds, bev_nms_iou)
                    for img_id, preds in dets.items()
                }

            results.update(dets)
            progress_bar.update()

        print("inference on {} images by {}/per image".format(
            len(self.dataloader), model_infer_time / len(self.dataloader)))

        progress_bar.close()

        # save the result for evaluation.
        self.logger.info('==> Saving ...')
        self.save_results(results)

    def save_results(self, results):
        self.output_dir = os.environ.get("OUTPUT_DIR", self.output_dir)
        output_dir = os.path.join(self.output_dir, 'outputs', 'data')
        os.makedirs(output_dir, exist_ok=True)

        for img_id in results.keys():
            if self.dataset_type == 'KITTI':
                output_path = os.path.join(output_dir, '{:06d}.txt'.format(img_id))
            else:
                os.makedirs(os.path.join(output_dir, self.dataloader.dataset.get_sensor_modality(img_id)), exist_ok=True)
                output_path = os.path.join(output_dir,
                                           self.dataloader.dataset.get_sensor_modality(img_id),
                                           self.dataloader.dataset.get_sample_token(img_id) + '.txt')

            f = open(output_path, 'w')
            for i in range(len(results[img_id])):
                class_name = self.class_name[int(results[img_id][i][0])]
                f.write('{} 0.0 0'.format(class_name))
                for j in range(1, len(results[img_id][i])):
                    f.write(' {:.2f}'.format(results[img_id][i][j]))
                f.write('\n')
            f.close()
    
        
    def prepare_targets(self, targets, batch_size):
        targets_list = []
        mask = targets['mask_2d']

        key_list = ['labels', 'boxes', 'calibs', 'depth', 'size_3d', 'heading_bin', 'heading_res', 'boxes_3d', 'bev_center', 'bev_gt_map']
        for bz in range(batch_size):
            target_dict = {}
            for key, val in targets.items():
                if key in key_list:
                    target_dict[key] = val[bz][mask[bz]]
                if key == 'depth_map':
                    target_dict[key] = val[bz]
                if key == 'obj_region':
                    target_dict[key] = val[bz]
                if key == 'bev_gt_map':
                    target_dict[key] = val[bz]
            targets_list.append(target_dict)
        return targets_list

    def evaluate(self):
        if not self._should_evaluate():
            self.logger.info("==> No local GT available for split 'test'; nothing to evaluate.")
            return None
        results_dir = os.path.join(self.output_dir, 'outputs', 'data')
        assert os.path.exists(results_dir)
        result = self.dataloader.dataset.eval(results_dir=results_dir, logger=self.logger)
        return result
