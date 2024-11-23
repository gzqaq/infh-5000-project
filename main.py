from pathlib import Path

from mmengine.config import Config
from mmengine.dataset import Compose
from mmengine.runner import Runner

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
