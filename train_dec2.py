import cv2
import numpy as np
import torch
from detectron2.data import DatasetMapper
from detectron2.data import build_detection_test_loader
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import transforms as T

setup_logger()

# import some common libraries
import os

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
import pickle
from utils import save_preview_input_gt_output, apply_lut_mask
from trainer_dec2 import AsbestosTrainer

TRAIN = False
EVALUATE = False
EVALUATE_ASBESTOS_CLASSIFICATION = True
GENERATE_EVAL_IMAGES = False
EXPERIMENT = "output_experiment46"
OUTPUT = f"/home/kevinmf94/{EXPERIMENT}"

print("CUDA ON: " + str(torch.cuda.is_available()))

def read_dicts(data_set):
    with (open(f'dav_dataset/{data_set}_dec.pkl', "rb")) as file:
        print("LOAD DICT " + data_set)
        return list(pickle.load(file))


for d in ["train", "val", "test"]:
    DatasetCatalog.register("data_" + d, lambda d=d: read_dicts(d))
    MetadataCatalog.get("data_" + d).set(thing_classes=["asbestos", "builds"])

cfg = get_cfg()
# cfg.MODEL.DEVICE = "cpu"
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))

cfg.DATASETS.TRAIN = ("data_train",)
cfg.DATASETS.TEST = ("data_val",)
cfg.TEST.EVAL_PERIOD = 200
cfg.DATALOADER.NUM_WORKERS = 4

# Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 45  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.001  # pick a good LR
cfg.SOLVER.MAX_ITER = 6000  # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []  # do not decay learning rate
cfg.SOLVER.CHECKPOINT_PERIOD = 1000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256  # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)

# Quantity of classes
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 2

# Load Pretrained weights:
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

print(f"LR={cfg.SOLVER.BASE_LR} MAX_ITER={cfg.SOLVER.MAX_ITER} BATCH={cfg.SOLVER.IMS_PER_BATCH}")

cfg.INPUT.MASK_FORMAT = "bitmask"

cfg.OUTPUT_DIR = OUTPUT
print(cfg.OUTPUT_DIR)
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if TRAIN:
    trainer = AsbestosTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

print("-------- Loading weights --------------")
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
print("-------- Loadeded ---------------------")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
predictor = DefaultPredictor(cfg)

if GENERATE_EVAL_IMAGES:

    print("-------- GENERATE_EVAL_IMAGES --------------")
    dataset = DatasetCatalog.get('data_test')
    metadata = MetadataCatalog.get('data_test')

    count = 1
    for d in dataset:
        print(f"    \rImage {count}", end=' ')
        im = cv2.imread(d["file_name"])
        mask = cv2.imread(d['file_name_mask'], -1).astype(np.uint8)
        mask = apply_lut_mask(mask)

        outputs = predictor(
            im)  # format is documented at https://detectron2.readthedocs.io/tutorials/models.html#model-output-format
        v = Visualizer(im[:, :, ::-1],
                       metadata=metadata,
                       instance_mode=ColorMode.IMAGE_BW,
                       scale=0.7
                       # remove the colors of unsegmented pixels. This option is only available for segmentation models
                       )
        if len(outputs["instances"]) > 0:
            out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
            save_preview_input_gt_output(f'{EXPERIMENT}/{d["image_id"]}_out.jpg', im, mask, out.get_image()[:, :, ::-1])

        count = count + 1

    print("---------------------------------------------")

if EVALUATE:
    print("-------- COCO RESULTS EVALUATE --------------")
    evaluator = COCOEvaluator("data_test", output_dir=cfg.OUTPUT_DIR)
    mapper = DatasetMapper(cfg, is_train=False, augmentations=[
        T.RandomFlip(prob=0.5, horizontal=False, vertical=True),
        T.RandomFlip(prob=0.5, horizontal=True, vertical=False)
    ])

    val_loader = build_detection_test_loader(cfg, "data_test", mapper=mapper)
    print(inference_on_dataset(predictor.model, val_loader, evaluator))
    print("---------------------------------------------")

if EVALUATE_ASBESTOS_CLASSIFICATION:

    print("------------------ EVALUATE_ASBESTOS_CLASSIFICATION -----------------", end='\n\r\n')
    test = DatasetCatalog.get('data_test')
    metadata = MetadataCatalog.get('data_test')

    result = []
    gt = []
    count = 1
    for d in test:
        print(f"    \rImage {count}", end=' ')
        im = cv2.imread(d["file_name"])
        # mask = cv2.imread(d['file_name_mask'], -1).astype(np.uint8)

        outputs = predictor(im)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy()
        scores = instances.scores.numpy()
        classes = classes[scores > 0.5]

        if len(classes) > 0 and np.any(classes == 0):
            result.append(0)
        else:
            result.append(1)

        gt.append(d['class'] - 1)
        count = count + 1

    result = np.array(result)
    gt = np.array(gt)

    TP = np.sum(np.logical_and(result == 0, gt == 0))
    FN = np.sum(np.logical_and(result == 1, gt == 0))

    FP = np.sum(np.logical_and(result == 0, gt == 1))
    TN = np.sum(np.logical_and(result == 1, gt == 1))
    total = len(gt)
    print(f"[TP: {TP}, FN: {FN}], \n\r[FP: {FP}, TN: {TN}], \n\rTOTAL: {total}")
    print(
        f"Classification Asbestos vs Non-Asbestos\n\r Accuracy: {(TP + TN) / total} " \
        + f"\n\r TPR: {TP / (TP + FN + 0.00001)}")
    print(f"----------------------------------------------------------------")
