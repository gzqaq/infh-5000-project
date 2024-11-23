from pathlib import Path

from mmengine.config import Config, DictAction
from mmengine.dataset import Compose
from mmengine.runner import Runner
from mmyolo.registry import RUNNERS

WORK_DIR = Path(__file__).parent.resolve()
CONF_DIR = WORK_DIR / "configs"
CKPT_DIR = WORK_DIR / "ckpts"

config = Config.fromfile(CONF_DIR / "hf_app.py")
config.load_from = str(
    CKPT_DIR / "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth"
)

print("\x1b[1;32mSuccessfully load config!\x1b[0m")
