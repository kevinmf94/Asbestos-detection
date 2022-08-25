import copy
import itertools
from typing import Optional

import torch
import datetime
import logging
import time
import os
import numpy as np

from detectron2.data import DatasetMapper, DatasetCatalog
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.data.samplers import TrainingSampler
from detectron2.data.transforms import Augmentation
from detectron2.evaluation import COCOEvaluator
from detectron2.utils import comm
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import transforms as T
from detectron2.engine import DefaultTrainer, HookBase
import detectron2.data.detection_utils as utils
from fvcore.transforms.transform import Transform, NoOpTransform


from torch.utils.data.sampler import WeightedRandomSampler


class AlbumentationsTransform(Transform):
    def __init__(self, aug, params):
        self.aug = aug
        self.params = params

    def apply_coords(self, coords: np.ndarray) -> np.ndarray:
        return coords

    def apply_image(self, image):
        return self.aug.apply(image, **self.params)

    def apply_box(self, box: np.ndarray) -> np.ndarray:
        try:
            return np.array(self.aug.apply_to_bboxes(box.tolist(), **self.params))
        except AttributeError:
            return box

    def apply_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        try:
            return self.aug.apply_to_mask(segmentation, **self.params)
        except AttributeError:
            return segmentation


class AlbumentationsWrapper(Augmentation):
    """
    Wrap an augmentor form the albumentations library: https://github.com/albu/albumentations.
    Image, Bounding Box and Segmentation are supported.
    Example:
    .. code-block:: python
        import albumentations as A
        from detectron2.data import transforms as T
        from detectron2.data.transforms.albumentations import AlbumentationsWrapper

        augs = T.AugmentationList([
            AlbumentationsWrapper(A.RandomCrop(width=256, height=256)),
            AlbumentationsWrapper(A.HorizontalFlip(p=1)),
            AlbumentationsWrapper(A.RandomBrightnessContrast(p=1)),
        ])  # type: T.Augmentation

        # Transform XYXY_ABS -> XYXY_REL
        h, w, _ = IMAGE.shape
        bbox = np.array(BBOX_XYXY) / [w, h, w, h]

        # Define the augmentation input ("image" required, others optional):
        input = T.AugInput(IMAGE, boxes=bbox, sem_seg=IMAGE_MASK)

        # Apply the augmentation:
        transform = augs(input)
        image_transformed = input.image  # new image
        sem_seg_transformed = input.sem_seg  # new semantic segmentation
        bbox_transformed = input.boxes   # new bounding boxes

        # Transform XYXY_REL -> XYXY_ABS
        h, w, _ = image_transformed.shape
        bbox_transformed = bbox_transformed * [w, h, w, h]
    """

    def __init__(self, augmentor):
        """
        Args:
            augmentor (albumentations.BasicTransform):
        """
        # super(Albumentations, self).__init__() - using python > 3.7 no need to call rng
        self._aug = augmentor

    def get_transform(self, image):
        do = self._rand_range() < self._aug.p
        if do:
            params = self.prepare_param(image)
            return AlbumentationsTransform(self._aug, params)
        else:
            return NoOpTransform()

    def prepare_param(self, image):
        params = self._aug.get_params()
        if self._aug.targets_as_params:
            targets_as_params = {"image": image}
            params_dependent_on_targets = self._aug.get_params_dependent_on_targets(targets_as_params)
            params.update(params_dependent_on_targets)
        params = self._aug.update_params(params, **{"image": image})
        return params


class LossEvalHook(HookBase):
    def __init__(self, eval_period, model, data_loader):
        self._model = model
        self._period = eval_period
        self._data_loader = data_loader

    def _do_loss_eval(self):
        # Copying inference_on_dataset from evaluator.py
        total = len(self._data_loader)
        num_warmup = min(5, total - 1)

        start_time = time.perf_counter()
        total_compute_time = 0
        losses = []
        for idx, inputs in enumerate(self._data_loader):
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_compute_time = 0
            start_compute_time = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            seconds_per_img = total_compute_time / iters_after_start
            if idx >= num_warmup * 2 or seconds_per_img > 5:
                total_seconds_per_img = (time.perf_counter() - start_time) / iters_after_start
                eta = datetime.timedelta(seconds=int(total_seconds_per_img * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    "Loss on Validation  done {}/{}. {:.4f} s / img. ETA={}".format(
                        idx + 1, total, seconds_per_img, str(eta)
                    ),
                    n=5,
                )
            loss_batch = self._get_loss(inputs)
            losses.append(loss_batch)
        mean_loss = np.mean(losses)

        self.trainer.storage.put_scalar('total_loss/validation_loss', mean_loss)
        comm.synchronize()

        return losses

    def _get_loss(self, data):
        # How loss is calculated on train_loop
        metrics_dict = self._model(data)
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        total_losses_reduced = sum(loss for loss in metrics_dict.values())
        return total_losses_reduced

    def after_step(self):
        next_iter = self.trainer.iter + 1
        is_final = next_iter == self.trainer.max_iter
        if is_final or (self._period > 0 and next_iter % self._period == 0):
            self._do_loss_eval()
        self.trainer.storage.put_scalars(timetest=12)


class AsbestosMapper(DatasetMapper):

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        if dataset_dict['class'] == 1:
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms = self.augmentations(aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            if dataset_dict['class'] == 1:
                self._transform_annotations(dataset_dict, transforms, image_shape)
            else:
                self._transform_annotations(dataset_dict, [], image_shape)

        return dataset_dict


class AsbestosWeightedSampler(TrainingSampler):

    def __init__(self, cfg, dataset, weights, size: int, shuffle: bool = True, seed: Optional[int] = None):
        super().__init__(size, shuffle, seed)
        self.dataset = dataset
        self.cfg = cfg
        self.weights = weights

        asb = 0
        non = 0
        for i in self.generate_samples():
            if self.dataset[i]['class'] == 1:
                asb += 1
            else:
                non += 1

        print(f"Asb {asb} Non: {non}")

    def generate_samples(self):
        samples_weight = np.array([self.weights[item['class'] - 1] for item in self.dataset])
        samples_weight = torch.from_numpy(samples_weight)
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        return sampler

    def __iter__(self):
        yield from itertools.islice(self._infinite_indices(), self._rank, None, self._world_size)

    def _infinite_indices(self):
        while True:
            yield from self.generate_samples()


class AsbestosTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        dataset = DatasetCatalog.get(cfg.DATASETS.TRAIN[0])

        print(cfg.AUG_ASB)
        mapper = AsbestosMapper(cfg, is_train=True, augmentations=cfg.AUG_ASB)

        if cfg.WEIGHTED is not None:
            sampler = AsbestosWeightedSampler(cfg, dataset, cfg.WEIGHTED, len(dataset))
            return build_detection_train_loader(cfg, mapper=mapper, sampler=sampler)

        return build_detection_train_loader(cfg, mapper=mapper)


    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[
            T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
            T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
        ])

        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            os.makedirs(f"{cfg.OUTPUT_DIR}/coco_eval", exist_ok=True)
            output_folder = f"{cfg.OUTPUT_DIR}/coco_eval"

        return COCOEvaluator(dataset_name, cfg, False, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(-1, LossEvalHook(
            self.cfg.TEST.EVAL_PERIOD,
            self.model,
            build_detection_test_loader(
                self.cfg,
                self.cfg.DATASETS.TEST[0],
                DatasetMapper(self.cfg, is_train=True, augmentations=[
                    T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
                    T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
                ])
            )
        ))
        return hooks
