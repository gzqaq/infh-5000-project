from pathlib import Path

import torch
from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
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
