from pathlib import Path

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner

YOLO_DIR = Path(__file__).parent.resolve()
CONF_DIR = YOLO_DIR / "configs"
CKPT_DIR = YOLO_DIR.parent.parent / "ckpts"
HF_CKPT = "yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth"


def init_runner() -> Runner:
    config = Config.fromfile(CONF_DIR / "hf_app.py")
    config.load_from = f"{CKPT_DIR/HF_CKPT}"
    config.work_dir = "./wkdir"

    runner = Runner.from_cfg(config)
    runner.call_hook("before_run")
    runner.load_or_resume()
    runner.pipeline = Compose(config.test_dataloader.dataset.pipeline)
    runner.model.eval()

    return runner
