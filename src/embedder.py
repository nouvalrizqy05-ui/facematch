"""
src/embedder.py
===============
Ekstraktor embedding wajah menggunakan ArcFace via DeepFace.
"""

import os
import warnings
import numpy as np
import cv2
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore")


class FaceEmbedder:
    def __init__(self, detector_backend: str = "opencv"):
        self.detector_backend = detector_backend
        self._deepface = None
        self._initialized = False

    def _init_deepface(self):
        if self._initialized:
            return True
        try:
            from deepface import DeepFace
            self._deepface = DeepFace
            self._initialized = True
            return True
        except Exception:
            return False

    def get_embedding(
        self,
        image_bgr: np.ndarray,
        enforce_detection: bool = True,
    ) -> tuple:
        if not self._init_deepface():
            raise RuntimeError(
                "DeepFace tidak tersedia.\n"
                "Install dengan: pip install deepface tf-keras"
            )

        # Coba dengan enforce_detection=True dulu, fallback ke False
        for enforce in [True, False]:
            try:
                result = self._deepface.represent(
                    img_path=image_bgr,
                    model_name="ArcFace",
                    enforce_detection=enforce,
                    detector_backend=self.detector_backend,
                    align=True,
                )
                if not result:
                    continue

                emb = np.array(result[0]["embedding"], dtype=np.float32)
                emb = emb / (np.linalg.norm(emb) + 1e-10)
                face_crop = self._extract_crop(image_bgr, result[0])
                return emb, face_crop

            except Exception as e:
                error_msg = str(e).lower()
                if "face could not be detected" in error_msg or \
                   "no face" in error_msg or \
                   "cannot find" in error_msg:
                    if enforce:
                        continue  # coba fallback enforce=False
                    return None, None
                if not enforce:
                    raise RuntimeError(f"ArcFace error: {e}")
                continue

        return None, None

    def get_embedding_from_path(self, image_path: str) -> tuple:
        if not self._init_deepface():
            raise RuntimeError("DeepFace tidak tersedia.")
        try:
            result = self._deepface.represent(
                img_path=image_path,
                model_name="ArcFace",
                enforce_detection=False,
                detector_backend=self.detector_backend,
                align=True,
            )
            if not result:
                return None, None
            emb = np.array(result[0]["embedding"], dtype=np.float32)
            emb = emb / (np.linalg.norm(emb) + 1e-10)
            img = cv2.imread(image_path)
            face_crop = self._extract_crop(img, result[0]) if img is not None else None
            return emb, face_crop
        except Exception as e:
            if "face could not be detected" in str(e).lower():
                return None, None
            raise

    def _extract_crop(self, image_bgr: np.ndarray, result_dict: dict):
        try:
            fa = result_dict.get("facial_area", {})
            x = fa.get("x", 0)
            y = fa.get("y", 0)
            w = fa.get("w", image_bgr.shape[1])
            h = fa.get("h", image_bgr.shape[0])
            pad = int(min(w, h) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(image_bgr.shape[1], x + w + pad)
            y2 = min(image_bgr.shape[0], y + h + pad)
            crop_bgr = image_bgr[y1:y2, x1:x2]
            if crop_bgr.size == 0:
                return None
            crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            return cv2.resize(crop_rgb, (112, 112))
        except Exception:
            return None

    def is_available(self) -> bool:
        return self._init_deepface()
