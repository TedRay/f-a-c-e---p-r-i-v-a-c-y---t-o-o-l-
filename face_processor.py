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
from typing import List, Tuple, Optional
import base64

class FaceProcessor:
    """人脸检测与隐私处理核心类"""
    
    def __init__(self, models_dir: str = None):
        """初始化人脸检测器"""
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        # 使用 OpenCV DNN 人脸检测
        proto_path = models_dir / "deploy.prototxt"
        model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        
        # 下载模型文件（如果不存在）
        if not proto_path.exists() or not model_path.exists():
            self._download_models(models_dir)
        
        # 加载 DNN 模型
        self.dnn_net = None
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            print("✓ DNN 人脸检测器加载成功")
        except Exception as e:
            print(f"DNN 模型加载失败: {e}")
        
        # 同时加载 Haar 级联作为补充
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.haar_cascade_alt = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml'
        )
        print("✓ Haar 级联检测器加载成功")
    
    def _download_models(self, models_dir: Path):
        """下载人脸检测模型"""
        import urllib.request
        
        # Caffe 模型文件 URLs
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        
        print("正在下载人脸检测模型...")
        try:
            urllib.request.urlretrieve(proto_url, str(models_dir / "deploy.prototxt"))
            urllib.request.urlretrieve(model_url, str(models_dir / "res10_300x300_ssd_iter_140000.caffemodel"))
            print("✓ 模型下载完成")
        except Exception as e:
            print(f"模型下载失败: {e}")
    
    def _expand_face_box(self, x, y, w, h, img_h, img_w, padding=0.3):
        """扩展人脸边界框，确保完全覆盖"""
        pad_w = int(w * padding)
        pad_h = int(h * padding)
        
        new_x = max(0, x - pad_w)
        new_y = max(0, y - pad_h)
        new_w = min(img_w - new_x, w + 2 * pad_w)
        new_h = min(img_h - new_y, h + 2 * pad_h)
        
        return new_x, new_y, new_w, new_h
    
    def _merge_faces(self, faces, iou_threshold=0.3):
        """合并重叠的人脸框（非极大值抑制）"""
        if not faces:
            return faces
        
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in faces])
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # 按面积排序（大到小）
        order = areas.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            
            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]
        
        return [faces[i] for i in keep]
    
    def detect_faces(self, image: np.ndarray, confidence: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸（多方法融合，提高检出率）
        
        Args:
            image: 输入图片 (BGR 格式)
            confidence: 检测置信度阈值
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        h, w = image.shape[:2]
        all_faces = []
        
        # 方法1: DNN 检测
        if self.dnn_net is not None:
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
            self.dnn_net.setInput(blob)
            detections = self.dnn_net.forward()
            
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    fw, fh = endX - startX, endY - startY
                    if fw > 10 and fh > 10:  # 过滤太小的框
                        all_faces.append((startX, startY, fw, fh))
        
        # 方法2: Haar 级联检测（标准参数）
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 使用多组参数提高检出率
        haar_params = [
            # (scaleFactor, minNeighbors, minSize)
            (1.05, 3, (20, 20)),   # 宽松：适合小人脸
            (1.1, 4, (30, 30)),    # 中等
            (1.08, 3, (25, 25)),   # 中等偏宽松
        ]
        
        for cascade in [self.haar_cascade, self.haar_cascade_alt]:
            for (scale, neighbors, min_size) in haar_params:
                detected = cascade.detectMultiScale(gray, scaleFactor=scale, 
                                                     minNeighbors=neighbors, minSize=min_size)
                for (fx, fy, fw, fh) in detected:
                    fx, fy, fw, fh = int(fx), int(fy), int(fw), int(fh)
                    if fw > 10 and fh > 10:
                        all_faces.append((fx, fy, fw, fh))
        
        # 去重：合并重叠的人脸框
        unique_faces = self._merge_faces(all_faces, iou_threshold=0.3)
        
        # 扩展边界框，确保完全覆盖人脸
        expanded = []
        for (fx, fy, fw, fh) in unique_faces:
            ex, ey, ew, eh = self._expand_face_box(fx, fy, fw, fh, h, w, padding=0.25)
            expanded.append((ex, ey, ew, eh))
        
        return expanded
    
    def apply_blur(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """
        应用模糊平滑效果
        """
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        # 双边滤波 + 高斯模糊
        blurred = cv2.bilateralFilter(face_region, 15, 80, 80)
        blurred = cv2.GaussianBlur(blurred, (25, 25), 30)
        
        result = image.copy()
        result[y:y+h, x:x+w] = blurred
        return result
    
    def apply_pixelate(self, image: np.ndarray, face: Tuple[int, int, int, int], 
                       block_size: int = 15) -> np.ndarray:
        """
        应用像素化效果
        """
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        small = cv2.resize(face_region, (max(1, w // block_size), max(1, h // block_size)), 
                          interpolation=cv2.INTER_LINEAR)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        result = image.copy()
        result[y:y+h, x:x+w] = pixelated
        return result
    
    def apply_emoji(self, image: np.ndarray, face: Tuple[int, int, int, int], 
                   emoji_path: str = None) -> np.ndarray:
        """
        应用表情包贴纸效果
        """
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        result = image.copy()
        
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        
        # 黄色圆形
        cv2.circle(result, center, radius, (52, 213, 255), -1)
        cv2.circle(result, center, radius, (0, 0, 0), 3)
        
        # 眼睛
        eye_y = center[1] - radius // 4
        eye_offset = radius // 3
        cv2.circle(result, (center[0] - eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        cv2.circle(result, (center[0] + eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        
        # 笑脸
        smile_y = center[1] + radius // 4
        smile_radius = radius // 2
        cv2.ellipse(result, (center[0], smile_y), (smile_radius, smile_radius // 2), 
                    0, 0, 180, (0, 0, 0), 3)
        
        return result
    
    def process_image(self, image: np.ndarray, effect: str = "blur", 
                      whitelist_faces: List[Tuple[int, int, int, int]] = None,
                      whitelist_embeddings: List[np.ndarray] = None) -> np.ndarray:
        """
        处理单张图片，对非白名单人脸应用隐私保护效果
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return image
        
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face
            
            # 检查是否在白名单中
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
            
            # 应用隐私保护效果
            if effect == "blur":
                result = self.apply_blur(result, face)
            elif effect == "pixelate":
                result = self.apply_pixelate(result, face)
            elif effect == "emoji":
                result = self.apply_emoji(result, face)
        
        return result


def load_known_faces(known_faces_dir: str) -> Tuple[List[np.ndarray], List[Tuple[int, int, int, int]]]:
    """
    加载白名单人脸
    """
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
    """
    批量处理图片
    """
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
