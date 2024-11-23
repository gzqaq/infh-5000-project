from pathlib import Path

import cv2
import numpy as np
import supervision as sv
import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from PIL import Image
from torchvision.ops import nms

WORK_DIR = Path(__file__).parent.resolve()
CONF_DIR = WORK_DIR / "configs"
CKPT_DIR = WORK_DIR / "ckpts"

config = Config.fromfile(CONF_DIR / "hf_app.py")
config.load_from = str(
    CKPT_DIR / "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth"
)
config.work_dir = "./wkdir"

print("\x1b[1;32mSuccessfully load config!\x1b[0m")

runner = Runner.from_cfg(config)
runner.call_hook("before_run")
runner.load_or_resume()
runner.pipeline = Compose(config.test_dataloader.dataset.pipeline)
runner.model.eval()

print("\x1b[1;32mSuccessfully load runner!\x1b[0m")

# inputs
IMG_PATH = WORK_DIR / "wkdir" / "test-img.jpg"
TEXT = "woman,man"
NMS_THRES = 0.5
SCORE_THRES = 0.05
MAX_NUM_BOXES = 100

texts = [[t.strip()] for t in TEXT.split(",")]
data_info = runner.pipeline({"img_id": 0, "img_path": IMG_PATH, "texts": texts})
data_batch = {
    "inputs": data_info["inputs"][None],
    "data_samples": [data_info["data_samples"]],
}

with autocast(enabled=False), torch.no_grad():
    output = runner.model.test_step(data_batch)[0]

pred_instances = output.pred_instances
keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=NMS_THRES)
pred_instances = pred_instances[keep]
pred_instances = pred_instances[pred_instances.scores.float() > SCORE_THRES]

if len(pred_instances.scores) > MAX_NUM_BOXES:
    indices = pred_instances.scores.float().topk(MAX_NUM_BOXES)[1]
    pred_instances = pred_instances[indices]

pred_instances = pred_instances.cpu().numpy()

print(pred_instances)
print("\x1b[1;32mSuccessfully run image demo!\x1b[0m")

#+begin visualize for debug
detections = sv.Detections(
    xyxy=pred_instances["bboxes"],
    class_id=pred_instances["labels"],
    confidence=pred_instances["scores"],
)
labels = [
    f"{texts[class_id][0]} {confidence:0.2f}"
    for class_id, confidence in zip(detections.class_id, detections.confidence)
]
img = np.array(Image.open(IMG_PATH))
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
img = sv.BoundingBoxAnnotator().annotate(img, detections)
img = sv.LabelAnnotator(text_color=sv.Color.BLACK).annotate(
    img, detections, labels=labels
)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img)
# with open(IMG_PATH.with_suffix(".labeled.jpg"), "wb") as fd:
#     img.save(fd)
# print("\x1b[1;32mSuccessfully save demo image!\x1b[0m")
#+end


def mask_from_bbox(xyxy: np.ndarray, img: np.ndarray) -> np.ndarray:
    xyxy = np.round(xyxy).astype(np.int32)
    width = xyxy[2] - xyxy[0]
    height = xyxy[3] - xyxy[1]
    mask = np.zeros_like(img)
    mask[xyxy[1] :, xyxy[0] :][:height, :width] = 1

    return mask
