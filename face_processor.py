#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幸福的照片隐私打码工具
用途：森林疗愈活动照片批量隐私保护处理
使用 InsightFace (RetinaFace + ArcFace) 实现高精度检测与识别
"""

import cv2
import numpy as np
import os
from pathlib import Path
from typing import List, Tuple, Optional
from insightface.app import FaceAnalysis


class FaceProcessor:
    """人脸检测与隐私处理核心类（InsightFace 版）"""
    
    # ArcFace 特征匹配阈值（余弦相似度）
    # 同一人通常 > 0.4，不同人通常 < 0.3
    WHITELIST_THRESHOLD = 0.35
    
    def __init__(self, models_dir: str = None):
        """初始化 InsightFace"""
        self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=0, det_size=(640, 640))
        self.app.det_model.det_thresh = 0.5  # 默认检测阈值
        print("✓ InsightFace (RetinaFace + ArcFace) 初始化完成")
    
    def detect_faces(self, image: np.ndarray, sensitivity: str = "medium") -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸
        
        Args:
            image: 输入图片 (BGR 格式)
            sensitivity: 'low' / 'medium' / 'high'
            
        Returns:
            人脸边界框列表 [(x, y, w, h), ...]
        """
        # 设置检测阈值
        thresh_map = {"low": 0.6, "medium": 0.4, "high": 0.25}
        self.app.det_model.det_thresh = thresh_map.get(sensitivity, 0.4)
        
        faces = self.app.get(image)
        
        result = []
        for face in faces:
            bbox = face.bbox.astype(int)
            x, y = max(0, bbox[0]), max(0, bbox[1])
            w = bbox[2] - x
            h = bbox[3] - y
            
            if w > 15 and h > 15 and 0.4 < w/h < 2.5:
                # 扩展 15% 确保完全覆盖
                pad_x = int(w * 0.15)
                pad_y = int(h * 0.15)
                x = max(0, x - pad_x)
                y = max(0, y - pad_y)
                w = min(image.shape[1] - x, w + 2 * pad_x)
                h = min(image.shape[0] - y, h + 2 * pad_y)
                result.append((x, y, w, h))
        
        return result
    
    def get_face_embedding(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        获取人脸特征向量（使用 InsightFace 内置的 ArcFace）
        
        Args:
            image: 原始图片
            face: 人脸边界框 (x, y, w, h)
            
        Returns:
            512 维特征向量，失败返回 None
        """
        x, y, w, h = face
        # InsightFace 已经在 get() 时提取了 embedding
        # 这里重新检测该区域来获取特征
        faces = self.app.get(image)
        
        for f in faces:
            bbox = f.bbox.astype(int)
            fx, fy = bbox[0], bbox[1]
            fw, fh = bbox[2] - fx, bbox[3] - fy
            
            # 匹配到对应的人脸框（中心点距离足够近）
            face_cx, face_cy = x + w//2, y + h//2
            f_cx, f_cy = fx + fw//2, fy + fh//2
            dist = np.sqrt((face_cx - f_cx)**2 + (face_cy - f_cy)**2)
            
            if dist < max(w, h) * 0.5 and f.embedding is not None:
                return f.embedding
        
        return None
    
    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        计算两个特征的余弦相似度
        
        Returns:
            相似度 (0~1，越大越相似)
        """
        feat1 = feat1 / np.linalg.norm(feat1)
        feat2 = feat2 / np.linalg.norm(feat2)
        return float(np.dot(feat1, feat2))
    
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
                      whitelist_features: List[np.ndarray] = None,
                      sensitivity: str = "medium") -> np.ndarray:
        """
        处理单张图片
        
        Args:
            image: 输入图片
            effect: 效果类型 ('blur', 'pixelate', 'emoji')
            whitelist_features: 白名单人员特征向量列表
            sensitivity: 检测灵敏度
        """
        # 设置检测阈值
        thresh_map = {"low": 0.6, "medium": 0.4, "high": 0.25}
        self.app.det_model.det_thresh = thresh_map.get(sensitivity, 0.4)
        
        # InsightFace 一次调用同时获取检测框和特征
        insight_faces = self.app.get(image)
        
        if not insight_faces:
            return image
        
        result = image.copy()
        
        for face in insight_faces:
            bbox = face.bbox.astype(int)
            x, y = max(0, bbox[0]), max(0, bbox[1])
            w, h = bbox[2] - x, bbox[3] - y
            
            if w < 15 or h < 15:
                continue
            
            # 扩展边界框
            pad_x = int(w * 0.15)
            pad_y = int(h * 0.15)
            fx = max(0, x - pad_x)
            fy = max(0, y - pad_y)
            fw = min(image.shape[1] - fx, w + 2 * pad_x)
            fh = min(image.shape[0] - fy, h + 2 * pad_y)
            
            face_box = (fx, fy, fw, fh)
            
            # 白名单匹配
            if whitelist_features and face.embedding is not None:
                is_whitelisted = False
                for wl_feat in whitelist_features:
                    sim = self.compute_similarity(face.embedding, wl_feat)
                    if sim > self.WHITELIST_THRESHOLD:
                        is_whitelisted = True
                        break
                
                if is_whitelisted:
                    continue  # 跳过白名单人脸
            
            # 应用效果
            if effect == "blur":
                result = self.apply_blur(result, face_box)
            elif effect == "pixelate":
                result = self.apply_pixelate(result, face_box)
            elif effect == "emoji":
                result = self.apply_emoji(result, face_box)
        
        return result


def load_known_faces(known_faces_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    加载白名单人脸特征
    
    Returns:
        (人脸图片列表, 特征向量列表)
    """
    known_faces = []
    face_features = []
    
    app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))
    app.det_model.det_thresh = 0.5
    
    known_dir = Path(known_faces_dir)
    if not known_dir.exists():
        return known_faces, face_features
    
    for img_path in known_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        if img_path.name.startswith("_"):
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        faces = app.get(img)
        if not faces:
            continue
        
        # 取最大/最高置信度的人脸
        best = max(faces, key=lambda f: f.det_score)
        
        if best.embedding is not None:
            known_faces.append(img)
            face_features.append(best.embedding)
            print(f"✓ 白名单加载: {img_path.name}, 特征维度 {best.embedding.shape}")
        else:
            print(f"⚠ 白名单跳过: {img_path.name}, 特征提取失败")
    
    return known_faces, face_features


def process_batch(input_dir: str, output_dir: str, effect: str = "blur",
                  known_faces_dir: str = None) -> List[str]:
    """批量处理图片"""
    processor = FaceProcessor()
    
    whitelist_features = []
    if known_faces_dir and Path(known_faces_dir).exists():
        _, whitelist_features = load_known_faces(known_faces_dir)
    
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
        
        result = processor.process_image(img, effect, whitelist_features)
        
        output_file = output_path / f"processed_{img_path.name}"
        cv2.imwrite(str(output_file), result)
        processed.append(str(output_file))
    
    return processed


if __name__ == "__main__":
    processor = FaceProcessor()
    print("InsightFace 人脸处理器初始化完成")
