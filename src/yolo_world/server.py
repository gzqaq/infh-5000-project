import json
import logging
import time
from pathlib import Path
from tempfile import gettempdir

import numpy as np
import torch
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from PIL import Image
from torchvision.ops import nms

from src.communication.messages import YoloMessage
from src.yolo_world.init import init_runner
from src.yolo_world.utils import combine_masks, mask_from_box_coordinates


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
        self._log_path: Path
        self._logger: logging.Logger
        self._setup_logger()

        self.runner = init_runner()
        self._logger.info("Runner initialized.")

        self.nms_thres = nms_thres
        self.score_thres = score_thres
        self.max_num_boxes = max_num_boxes

        self.msg_file = Path(gettempdir()) / "yolo-world-server.msg"
        if not self.msg_file.exists():
            self.msg_file.touch()
        self._logger.info(f"Use {self.msg_file} to communicate.")

        self._timestamp: int
        self._update_timestamp()
        self._logger.info(f"Current timestamp: {self._timestamp}")

    def run(self) -> None:
        try:
            while True:
                msg = self._wait_for_msg()
                if msg.tgt_path is None:
                    save_path = msg.img_path.with_suffix(f".res{msg.img_path.suffix}")
                else:
                    save_path = msg.tgt_path

                self._logger.info(
                    f"Start inference on {msg.img_path} "
                    f"with labels {','.join(msg.labels)}."
                )
                boxes = self._inference(msg.img_path, msg.labels)
                self._logger.info(f"Detected {len(boxes)} objects.")

                # create mask and save masked image
                original_img = Image.open(msg.img_path)
                masks = [mask_from_box_coordinates(box, original_img) for box in boxes]
                if len(masks) == 0:
                    masks.append(np.ones_like(original_img))
                mask = combine_masks(masks)
                masked_img = original_img * mask
                self._logger.info("Masks successfully applied.")

                # call xmas hat
                self._logger.info("Christmas hats successfully added.")

                # overlay hatted image to original one
                overlayed_img = masked_img
                self._logger.info("Original image successfully overlayed.")

                # save processed image
                res_img = Image.fromarray(overlayed_img)
                with open(save_path, "wb") as fd:
                    res_img.save(fd)
                self._logger.info(f"Saved to {save_path}.")

        except KeyboardInterrupt:
            self._logger.warning("Abort due to keyboard interrupt.")  # TODO: use logger

    def _setup_logger(self) -> None:
        log_path = Path(gettempdir()) / "yolo-world-server.log"
        logger = logging.getLogger("yolo-world")
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter(
            "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"
        )

        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        self._log_path = log_path
        self._logger = logger

    def _update_timestamp(self) -> None:
        self._timestamp = self.msg_file.stat().st_mtime_ns
        self._logger.debug(f"Update timestamp to be {self._timestamp}.")

    def _wait_for_msg(self) -> YoloMessage:
        cnt = 0
        while True:
            if self.msg_file.stat().st_mtime_ns > self._timestamp:
                break
            else:
                time.sleep(0.1)
                cnt += 1
                if cnt >= 100:
                    self._logger.debug("No request for 10 seconds.")
                    cnt = 0

        self._update_timestamp()
        with open(self.msg_file, "r") as fd:
            return YoloMessage.from_dict(json.load(fd))

    def _inference(self, img_path: Path, labels: list[str]) -> np.ndarray:
        texts = [[label] for label in labels]
        data_info = self.runner.pipeline(
            {"img_id": 0, "img_path": img_path, "texts": texts}
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
            self._logger.info(
                f"Detect more objects than {self.max_num_boxes}. "
                f"Pick top {self.max_num_boxes}."
            )
            indices = results.scores.float().topk(self.max_num_boxes)[1]
            results = results[indices]

        results = results.cpu().numpy()
        return results.bboxes
