"""YOLO26n ONNX 추론 래퍼 — letterbox 전처리 + (1,300,6) NMS-free 파싱.

CLAUDE.md 10.1/10.2 실측:
    입력  images (1,3,640,640) float, RGB, 0..1, letterbox.
    출력  output0 (1,300,6), 6값 = [x1, y1, x2, y2, score, class_id].
          score 내림차순 정렬·NMS-free, 좌표는 letterbox된 640 스케일.
    → 앵커 디코딩/NMS 재적용 금지. 여기서 letterbox 역변환만 수행.

onnxruntime/cv2/numpy 는 보드(aarch64)에만 설치돼 있어(10.2), 무거운 import 는
지연(함수/생성자 내부)한다 — 개발 박스에서 driving_policy 등 순수 모듈을
import 할 때 이 파일이 딸려 import 돼도 깨지지 않게 하기 위함. 런타임/모델 부재
시 RuntimeError 로 명확히 실패시켜, inference_node 가 degraded(정지) 모드로
빠지게 한다.
"""
from dataclasses import dataclass


@dataclass
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def letterbox(img, new_shape=640, color=(114, 114, 114)):
    """비율 유지 리사이즈 + 중앙 패딩. 반환 (canvas, ratio, pad_left, pad_top)."""
    import cv2
    import numpy as np

    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((new_shape, new_shape, 3), color, dtype=np.uint8)
    top = (new_shape - nh) // 2
    left = (new_shape - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas, r, left, top


class YoloOnnx:
    def __init__(self, model_path, imgsz=640, conf_threshold=0.25,
                 providers=('CPUExecutionProvider',)):
        import os

        if not model_path or not os.path.exists(model_path):
            raise RuntimeError(f'ONNX 모델을 찾을 수 없음: {model_path}')
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - 보드에서만 존재
            raise RuntimeError(f'onnxruntime import 실패: {exc}')

        self.imgsz = int(imgsz)
        self.conf_threshold = float(conf_threshold)
        self.session = ort.InferenceSession(model_path, providers=list(providers))
        self.input_name = self.session.get_inputs()[0].name

    def infer(self, bgr_image):
        """BGR 프레임 → Detection 리스트(원본 이미지 좌표)."""
        import numpy as np

        canvas, r, left, top = letterbox(bgr_image, self.imgsz)
        # BGR→RGB, HWC→CHW, 0..1, (1,3,H,W)
        blob = canvas[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
        blob = np.ascontiguousarray(blob[None, ...])
        out = self.session.run(None, {self.input_name: blob})[0]
        return self._parse(out, r, left, top)

    def _parse(self, out, r, left, top):
        """(1,300,6) 또는 (300,6) → Detection 리스트. score 내림차순이라 첫
        conf 미만에서 break(뒤는 zero-padding)."""
        arr = out[0] if getattr(out, 'ndim', 2) == 3 else out
        dets = []
        for row in arr:
            score = float(row[4])
            if score < self.conf_threshold:
                break
            x1 = (float(row[0]) - left) / r
            y1 = (float(row[1]) - top) / r
            x2 = (float(row[2]) - left) / r
            y2 = (float(row[3]) - top) / r
            dets.append(Detection(int(round(float(row[5]))), score, x1, y1, x2, y2))
        return dets
