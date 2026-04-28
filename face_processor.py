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
        
        try:
            self.detector = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            self.use_dnn = True
            print("✓ DNN 人脸检测器加载成功")
        except Exception as e:
            print(f"DNN 模型加载失败: {e}，使用 Haar 级联")
            self.use_dnn = False
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.detector = cv2.CascadeClassifier(cascade_path)
    
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
    
    def detect_faces(self, image: np.ndarray, confidence: float = 0.5) -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸
        
        Args:
            image: 输入图片 (BGR 格式)
            confidence: 检测置信度阈值
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        h, w = image.shape[:2]
        faces = []
        
        if self.use_dnn:
            # DNN 方法
            blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104, 177, 123))
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            for i in range(detections.shape[2]):
                conf = detections[0, 0, i, 2]
                if conf > confidence:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # 确保边界框在图片范围内
                    startX, startY = max(0, startX), max(0, startY)
                    endX, endY = min(w, endX), min(h, endY)
                    faces.append((startX, startY, endX - startX, endY - startY))
        else:
            # Haar 级联方法
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            detected = self.detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            faces = [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in detected]
        
        return faces
    
    def apply_blur(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> np.ndarray:
        """
        应用模糊平滑效果
        
        Args:
            image: 输入图片
            face: 人脸边界框 (x, y, w, h)
            
        Returns:
            处理后的图片
        """
        x, y, w, h = face
        # 确保坐标有效
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        # 使用双边滤波保持边缘，同时模糊细节
        blurred = cv2.bilateralFilter(face_region, 15, 80, 80)
        # 多次模糊增强效果
        blurred = cv2.GaussianBlur(blurred, (25, 25), 30)
        
        result = image.copy()
        result[y:y+h, x:x+w] = blurred
        return result
    
    def apply_pixelate(self, image: np.ndarray, face: Tuple[int, int, int, int], 
                       block_size: int = 15) -> np.ndarray:
        """
        应用像素化效果
        
        Args:
            image: 输入图片
            face: 人脸边界框 (x, y, w, h)
            block_size: 像素块大小
            
        Returns:
            处理后的图片
        """
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        face_region = image[y:y+h, x:x+w]
        
        # 缩小再放大实现像素化
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
        
        Args:
            image: 输入图片
            face: 人脸边界框 (x, y, w, h)
            emoji_path: 表情包图片路径（可选）
            
        Returns:
            处理后的图片
        """
        x, y, w, h = face
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 0 or h <= 0:
            return image
        
        # 使用内置可爱表情（用文字绘制）
        result = image.copy()
        
        # 绘制彩色圆形背景
        center = (x + w // 2, y + h // 2)
        radius = max(w, h) // 2
        
        # 绘制黄色圆形
        cv2.circle(result, center, radius, (52, 213, 255), -1)  # BGR: 黄色
        cv2.circle(result, center, radius, (0, 0, 0), 3)  # 黑色边框
        
        # 绘制眼睛
        eye_y = center[1] - radius // 4
        eye_offset = radius // 3
        cv2.circle(result, (center[0] - eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        cv2.circle(result, (center[0] + eye_offset, eye_y), radius // 8, (0, 0, 0), -1)
        
        # 绘制笑脸
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
        
        Args:
            image: 输入图片
            effect: 效果类型 ('blur', 'pixelate', 'emoji')
            whitelist_faces: 白名单人脸边界框列表
            whitelist_embeddings: 白名单人脸特征向量列表
            
        Returns:
            处理后的图片
        """
        faces = self.detect_faces(image)
        
        if not faces:
            return image
        
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face
            
            # 检查是否在白名单中（简化版本：基于位置重叠）
            if whitelist_faces:
                is_whitelisted = False
                face_center = (x + w // 2, y + h // 2)
                
                for wl_face in whitelist_faces:
                    wl_x, wl_y, wl_w, wl_h = wl_face
                    wl_center = (wl_x + wl_w // 2, wl_y + wl_h // 2)
                    
                    # 计算中心点距离
                    dist = np.sqrt((face_center[0] - wl_center[0])**2 + (face_center[1] - wl_center[1])**2)
                    if dist < max(w, h) // 2:
                        is_whitelisted = True
                        break
                
                if is_whitelisted:
                    continue  # 跳过白名单人脸
            
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
    
    Args:
        known_faces_dir: 白名单人脸图片目录
        
    Returns:
        (人脸图片列表, 人脸边界框列表)
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
                    face_boxes.append(faces[0])  # 取第一张人脸
    
    return known_faces, face_boxes


def process_batch(input_dir: str, output_dir: str, effect: str = "blur",
                  known_faces_dir: str = None) -> List[str]:
    """
    批量处理图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        effect: 效果类型
        known_faces_dir: 白名单人脸目录
        
    Returns:
        处理成功的文件列表
    """
    processor = FaceProcessor()
    
    # 加载白名单
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
    # 测试代码
    processor = FaceProcessor()
    print("人脸处理器初始化完成")
