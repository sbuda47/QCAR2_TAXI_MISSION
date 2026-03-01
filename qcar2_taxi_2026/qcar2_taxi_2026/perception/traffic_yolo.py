# qcar2_taxi_2026/perception/traffic_yolo.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Detection:
    label: str
    conf: float
    # xyxy in pixel coordinates
    x1: float
    y1: float
    x2: float
    y2: float


class TrafficYolo:
    """
    YOLOv8 wrapper for traffic/sign detection.

    - If ultralytics isn't installed OR model_path is None -> runs in stub mode.
    - You can supply a custom model trained on:
        stop sign, yield sign, traffic light (red/yellow/green)
      or use separate classes (traffic_light_red, etc).

    Expected inference input: BGR image (OpenCV).
    Output: List[Detection]
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        conf_thres: float = 0.35,
        iou_thres: float = 0.45,
        device: str = "cpu",
    ):
        self.model_path = model_path
        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.device = device

        self._model = None
        self._names = None
        self._available = False

        if model_path:
            self._try_load()

    @property
    def available(self) -> bool:
        return self._available

    def _try_load(self) -> None:
        try:
            from ultralytics import YOLO  # type: ignore

            self._model = YOLO(self.model_path)
            # Some ultralytics versions have .names
            self._names = getattr(self._model, "names", None)
            self._available = True
        except Exception:
            # stay in stub mode
            self._model = None
            self._names = None
            self._available = False

    def infer(self, bgr: np.ndarray) -> List[Detection]:
        """
        Run YOLO inference on a BGR frame.

        Returns [] in stub mode.
        """
        if not self._available or self._model is None:
            return []

        # Ultralytics YOLO accepts numpy arrays (BGR/RGB both work; it converts internally)
        results = self._model.predict(
            source=bgr,
            conf=self.conf_thres,
            iou=self.iou_thres,
            device=self.device,
            verbose=False,
        )

        out: List[Detection] = []
        if not results:
            return out

        r0 = results[0]
        boxes = getattr(r0, "boxes", None)
        if boxes is None:
            return out

        # boxes.xyxy, boxes.conf, boxes.cls
        xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.array(boxes.xyxy)
        conf = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.array(boxes.conf)
        cls = boxes.cls.cpu().numpy() if hasattr(boxes.cls, "cpu") else np.array(boxes.cls)

        for i in range(len(xyxy)):
            c = float(conf[i])
            if c < self.conf_thres:
                continue
            cid = int(cls[i])
            label = str(cid)
            if isinstance(self._names, dict) and cid in self._names:
                label = str(self._names[cid])
            x1, y1, x2, y2 = map(float, xyxy[i])
            out.append(Detection(label=label, conf=c, x1=x1, y1=y1, x2=x2, y2=y2))

        return out