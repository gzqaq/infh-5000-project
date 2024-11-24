import json
import time
from pathlib import Path
from tempfile import mkstemp

import numpy as np
import torch
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from PIL import Image
from torchvision.ops import nms

from .init import init_runner


class Server:
    runner: Runner
    nms_thres: float
    score_thres: float
    max_num_boxes: int
    msg_file: Path

    def __init__(
        self,
        nms_thres: float = 0.5,
        score_thres: float = 0.05,
        max_num_boxes: int = 100,
    ) -> None:
        self.runner = init_runner()
        self.nms_thres = nms_thres
        self.score_thres = score_thres
        self.max_num_boxes = max_num_boxes
        self.msg_file = Path(mkstemp(prefix="yolo-world-server")[1])
        self._time_stamp = self.msg_file.stat().st_mtime_ns

    def run(self) -> None:
        try:
            while True:
                msg = self._wait_for_msg()

                # format input

                # inference

                # create mask and save masked image

                # call xmas hat

                # overlay hatted image to original one

                # save processed image

        except KeyboardInterrupt:
            print("Abort due to keyboard interrupt.")  # TODO: use logger

    def _wait_for_msg(self) -> dict[str, str]:  # TODO: return a dataclass
        while True:
            if self.msg_file.stat().st_mtime_ns > self._time_stamp:
                break
            else:
                time.sleep(0.1)

        with open(self.msg_file, "r") as fd:
            return json.load(fd)

    def _inference(self, img_path: Path, labels: list[str]) -> np.ndarray:
        data_info = self.runner.pipeline(
            {"img_id": 0, "img_path": img_path, "texts": labels}
        )
        data_batch = {
            "inputs": data_info["inputs"][None],
            "data_samples": [data_info["data_samples"]],
        }

        with autocast(enabled=False), torch.no_grad():
            output = self.runner.model.test_step(data_batch)[0]

        results = output.pred_instances
        keep = nms(results.bboxes, results.scores, iou_threshold=self.nms_thres)
        results = results[keep]
        results = results[results.scores.float() > self.score_thres]

        if len(results.scores) > self.max_num_boxes:
            indices = results.scores.float().topk(self.max_num_boxes)[1]
            results = results[indices]

        results = results.cpu().numpy()
        return results.bboxes
