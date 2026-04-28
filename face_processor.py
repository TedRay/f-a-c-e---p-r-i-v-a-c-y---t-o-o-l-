#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幸福的照片隐私打码工具
用途：森林疗愈活动照片批量隐私保护处理
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple

class FaceProcessor:
    """人脸检测与隐私处理核心类"""
    
    def __init__(self, models_dir: str = None):
        """初始化人脸检测器"""
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        proto_path = models_dir / "deploy.prototxt"
        model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        if not proto_path.exists() or not model_path.exists():
            self._download_models(models_dir)
        
        self.dnn_net = None
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            print("✓ DNN 人脸检测器加载成功")
        except Exception as e:
            print(f"DNN 模型加载失败: {e}")
        
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Haar 级联检测器加载成功")
    
    def _download_models(self, models_dir: Path):
        """下载人脸检测模型"""
        import urllib.request
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        print("正在下载人脸检测模型...")
        try:
            urllib.request.urlretrieve(proto_url, str(models_dir / "deploy.prototxt"))
            urllib.request.urlretrieve(model_url, str(models_dir / "res10_300x300_ssd_iter_140000.caffemodel"))
            print("✓ 模型下载完成")
        except Exception as e:
            print(f"模型下载失败: {e}")
    
    def _is_valid_face(self, x, y, w, h, img_h, img_w):
        """验证人脸框有效性"""
        # 尺寸限制
        if w < 25 or h < 25:
            return False
        if w > img_w * 0.6 or h > img_h * 0.6:
            return False
        # 宽高比限制（人脸接近正方形）
        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.0:
            return False
        # 位置限制
        if x < 0 or y < 0:
            return False
        if x + w > img_w or y + h > img_h:
            return False
        return True
    
    def _nms(self, faces_with_scores, iou_threshold=0.4):
        """非极大值抑制"""
        if len(faces_with_scores) <= 1:
            return [f for f, _ in faces_with_scores]
        
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h), _ in faces_with_scores])
        scores = np.array([s for _, s in faces_with_scores])
        
        x1, y1 = boxes[:, 0], boxes[:, 1]
        x2, y2 = boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        order = scores.argsort()[::-1]
        keep = []
        
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            if order.size == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            union = areas[i] + areas[order[1:]] - inter
            iou = inter / (union + 1e-6)
            
            order = order[np.where(iou <= iou_threshold)[0] + 1]
        
        return [faces_with_scores[i][0] for i in keep]
    
    def _expand_box(self, x, y, w, h, img_h, img_w, padding=0.15):
        """扩展人脸框"""
        px = int(w * padding)
        py = int(h * padding)
        nx = max(0, x - px)
        ny = max(0, y - py)
        nw = min(img_w - nx, w + 2 * px)
        nh = min(img_h - ny, h + 2 * py)
        return nx, ny, nw, nh
    
    def detect_faces(self, image: np.ndarray, sensitivity: str = "medium") -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸
        
        Args:
            image: 输入图片 (BGR 格式)
            sensitivity: 灵敏度 - 'low'(少检误检少), 'medium'(平衡), 'high'(多检可能误检)
        
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        h, w = image.shape[:2]
        all_faces = []
        
        # 灵敏度参数映射
        params = {
            "low": {
                "dnn_input": 300,
                "dnn_conf": 0.6,
                "haar_scale": 1.2,
                "haar_neighbors": 8,
                "haar_min_size": (40, 40),
                "iou_thresh": 0.3,
            },
            "medium": {
                "dnn_input": 500,
                "dnn_conf": 0.4,
                "haar_scale": 1.15,
                "haar_neighbors": 6,
                "haar_min_size": (35, 35),
                "iou_thresh": 0.4,
            },
            "high": {
                "dnn_input": 600,
                "dnn_conf": 0.25,
                "haar_scale": 1.08,
                "haar_neighbors": 4,
                "haar_min_size": (25, 25),
                "iou_thresh": 0.5,
            },
        }
        
        p = params.get(sensitivity, params["medium"])
        
        # === DNN 检测 ===
        if self.dnn_net is not None:
            input_size = p["dnn_input"]
            blob = cv2.dnn.blobFromImage(image, 1.0, (input_size, input_size), (104, 177, 123))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > p["dnn_conf"]:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (sx, sy, ex, ey) = box.astype("int")
                    sx, sy = max(0, sx), max(0, sy)
                    ex, ey = min(w, ex), min(h, ey)
                    fw, fh = ex - sx, ey - sy
                    
                    if self._is_valid_face(sx, sy, fw, fh, h, w):
                        all_faces.append(((sx, sy, fw, fh), float(conf)))
        
        # === Haar 补充检测 ===
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        detected = self.haar_cascade.detectMultiScale(
            gray,
            scaleFactor=p["haar_scale"],
            minNeighbors=p["haar_neighbors"],
            minSize=p["haar_min_size"],
            maxSize=(int(w * 0.4), int(h * 0.4))
        )
        
        for (fx, fy, fw, fh) in detected:
            if self._is_valid_face(int(fx), int(fy), int(fw), int(fh), h, w):
                all_faces.append(((int(fx), int(fy), int(fw), int(fh)), 0.35))
        
        # === NMS 去重 ===
        unique_faces = self._nms(all_faces, iou_threshold=p["iou_thresh"])
        
        # === 扩展边界框 ===
        expanded = []
        for (fx, fy, fw, fh) in unique_faces:
            ex, ey, ew, eh = self._expand_box(fx, fy, fw, fh, h, w, padding=0.15)
            expanded.append((ex, ey, ew, eh))
        
        return expanded
    
    def apply_blur(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """应用模糊平滑效果"""
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        blurred = cv2.bilateralFilter(face_region, 15, 80, 80)
        blurred = cv2.GaussianBlur(blurred, (25, 25), 30)
        
        result = image.copy()
        result[y:y+h, x:x+w] = blurred
        return result
    
    def apply_pixelate(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """应用像素化效果"""
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        block_size = max(10, min(w, h) // 10)
        small = cv2.resize(face_region, (max(1, w // block_size), max(1, h // block_size)))
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        result = image.copy()
        result[y:y+h, x:x+w] = pixelated
        return result
    
    def apply_emoji(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """应用表情包贴纸效果"""
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        result = image.copy()
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        
        cv2.circle(result, center, radius, (52, 213, 255), -1)
        cv2.circle(result, center, radius, (0, 0, 0), 3)
        
        eye_y = center[1] - radius // 4
        eye_offset = radius // 3
        cv2.circle(result, (center[0] - eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        cv2.circle(result, (center[0] + eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        
        smile_y = center[1] + radius // 4
        smile_radius = radius // 2
        cv2.ellipse(result, (center[0], smile_y), (smile_radius, smile_radius // 2), 0, 0, 180, (0, 0, 0), 3)
        
        return result
    
    def process_image(self, image: np.ndarray, effect: str = "blur", 
                      whitelist_faces: List[Tuple[int, int, int, int]] = None,
                      sensitivity: str = "medium") -> np.ndarray:
        """处理单张图片"""
        faces = self.detect_faces(image, sensitivity)
        
        if not faces:
            return image
        
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face
            
            if whitelist_faces:
                is_whitelisted = False
                face_center = (x + w // 2, y + h // 2)
                
                for wl_face in whitelist_faces:
                    wl_x, wl_y, wl_w, wl_h = wl_face
                    wl_center = (wl_x + wl_w // 2, wl_y + wl_h // 2)
                    dist = np.sqrt((face_center[0] - wl_center[0])**2 + (face_center[1] - wl_center[1])**2)
                    if dist < max(w, h) // 2:
                        is_whitelisted = True
                        break
                
                if is_whitelisted:
                    continue
            
            if effect == "blur":
                result = self.apply_blur(result, face)
            elif effect == "pixelate":
                result = self.apply_pixelate(result, face)
            elif effect == "emoji":
                result = self.apply_emoji(result, face)
        
        return result


def load_known_faces(known_faces_dir: str):
    """加载白名单人脸"""
    known_faces = []
    face_boxes = []
    
    processor = FaceProcessor()
    known_dir = Path(known_faces_dir)
    
    if not known_dir.exists():
        return known_faces, face_boxes
    
    for img_path in known_dir.glob("*"):
        if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
            img = cv2.imread(str(img_path))
            if img is not None:
                faces = processor.detect_faces(img)
                if faces:
                    known_faces.append(img)
                    face_boxes.append(faces[0])
    
    return known_faces, face_boxes


def process_batch(input_dir: str, output_dir: str, effect: str = "blur",
                  known_faces_dir: str = None) -> List[str]:
    """批量处理图片"""
    processor = FaceProcessor()
    
    whitelist_boxes = []
    if known_faces_dir and Path(known_faces_dir).exists():
        _, whitelist_boxes = load_known_faces(known_faces_dir)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    processed = []
    
    for img_path in input_path.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        result = processor.process_image(img, effect, whitelist_boxes)
        
        output_file = output_path / f"processed_{img_path.name}"
        cv2.imwrite(str(output_file), result)
        processed.append(str(output_file))
    
    return processed


if __name__ == "__main__":
    processor = FaceProcessor()
    print("人脸处理器初始化完成")
