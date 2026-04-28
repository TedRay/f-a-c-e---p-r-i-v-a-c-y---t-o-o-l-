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

class FaceProcessor:
    """人脸检测与隐私处理核心类"""
    
    # 白名单匹配相似度阈值（余弦相似度，越大越严格）
    WHITELIST_THRESHOLD = 0.28
    
    def __init__(self, models_dir: str = None):
        """初始化人脸检测器"""
        if models_dir is None:
            models_dir = Path(__file__).parent / "models"
        
        models_dir = Path(models_dir)
        models_dir.mkdir(exist_ok=True)
        
        self.models_dir = models_dir
        
        proto_path = models_dir / "deploy.prototxt"
        model_path = models_dir / "res10_300x300_ssd_iter_140000.caffemodel"
        face_rec_path = models_dir / "face_recognition_sface_2021dec.onnx"
        
        if not proto_path.exists() or not model_path.exists():
            self._download_models(models_dir)
        
        # DNN 人脸检测器
        self.dnn_net = None
        try:
            self.dnn_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))
            print("✓ DNN 人脸检测器加载成功")
        except Exception as e:
            print(f"DNN 模型加载失败: {e}")
        
        # Haar 级联
        self.haar_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        print("✓ Haar 级联检测器加载成功")
        
        # 人脸特征提取器（SFace）
        self.face_recognizer = None
        if face_rec_path.exists():
            try:
                self.face_recognizer = cv2.FaceRecognizerSF.create(
                    str(face_rec_path), ""
                )
                print("✓ SFace 人脸识别器加载成功")
            except Exception as e:
                print(f"SFace 模型加载失败: {e}")
        else:
            print("⚠ SFace 模型未找到，白名单将使用位置匹配")
    
    def _download_models(self, models_dir: Path):
        """下载人脸检测模型"""
        import urllib.request
        proto_url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        model_url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        face_rec_url = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        
        print("正在下载人脸检测模型...")
        try:
            urllib.request.urlretrieve(proto_url, str(models_dir / "deploy.prototxt"))
            urllib.request.urlretrieve(model_url, str(models_dir / "res10_300x300_ssd_iter_140000.caffemodel"))
            print("✓ 检测模型下载完成")
        except Exception as e:
            print(f"检测模型下载失败: {e}")
        
        print("正在下载人脸识别模型...")
        try:
            urllib.request.urlretrieve(face_rec_url, str(models_dir / "face_recognition_sface_2021dec.onnx"))
            print("✓ 识别模型下载完成")
        except Exception as e:
            print(f"识别模型下载失败: {e}")
    
    def _is_valid_face(self, x, y, w, h, img_h, img_w):
        """验证人脸框有效性"""
        if w < 25 or h < 25:
            return False
        if w > img_w * 0.6 or h > img_h * 0.6:
            return False
        aspect = w / h if h > 0 else 0
        if aspect < 0.5 or aspect > 2.0:
            return False
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
    
    def extract_face_feature(self, image: np.ndarray, face: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        提取人脸特征向量
        
        Args:
            image: 原始图片
            face: 人脸边界框 (x, y, w, h)
            
        Returns:
            128维特征向量，失败返回 None
        """
        if self.face_recognizer is None:
            return None
        
        x, y, w, h = face
        
        # 确保坐标有效
        x, y = max(0, x), max(0, y)
        w = min(w, image.shape[1] - x)
        h = min(h, image.shape[0] - y)
        
        if w <= 10 or h <= 10:
            return None
        
        try:
            # 裁剪人脸区域（多裁 20% 背景作为上下文）
            pad_x = int(w * 0.2)
            pad_y = int(h * 0.2)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(image.shape[1], x + w + pad_x)
            y2 = min(image.shape[0], y + h + pad_y)
            
            face_roi = image[y1:y2, x1:x2]
            if face_roi.size == 0:
                return None
            
            # 小人脸先放大再提取特征，提高质量
            roi_h, roi_w = face_roi.shape[:2]
            if roi_w < 100 or roi_h < 100:
                scale = max(2, 112 // min(roi_w, roi_h) + 1)
                face_roi = cv2.resize(face_roi, (roi_w * scale, roi_h * scale), 
                                      interpolation=cv2.INTER_CUBIC)
            
            # 缩放到 SFace 标准输入尺寸 (112x112)
            face_resized = cv2.resize(face_roi, (112, 112))
            
            # 轻微锐化提高细节
            sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            face_resized = cv2.filter2D(face_resized, -1, sharpen_kernel * 0.3 + np.eye(3) * 0.7)
            
            # 提取特征
            feature = self.face_recognizer.feature(face_resized)
            return feature
        except Exception as e:
            print(f"特征提取失败: {e}")
            return None
    
    def compute_similarity(self, feature1: np.ndarray, feature2: np.ndarray) -> float:
        """
        计算两个人脸特征的余弦相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            
        Returns:
            余弦相似度 (0~1，越大越相似)
        """
        # 使用 OpenCV 内置的距离计算
        score = self.face_recognizer.match(feature1, feature2, cv2.FaceRecognizerSF_FR_COSINE)
        return float(score)
    
    def detect_faces(self, image: np.ndarray, sensitivity: str = "medium") -> List[Tuple[int, int, int, int]]:
        """
        检测图片中的人脸
        """
        h, w = image.shape[:2]
        all_faces = []
        
        params = {
            "low": {
                "dnn_input": 300, "dnn_conf": 0.6,
                "haar_scale": 1.2, "haar_neighbors": 8, "haar_min_size": (40, 40),
                "iou_thresh": 0.3,
            },
            "medium": {
                "dnn_input": 500, "dnn_conf": 0.4,
                "haar_scale": 1.15, "haar_neighbors": 6, "haar_min_size": (35, 35),
                "iou_thresh": 0.4,
            },
            "high": {
                "dnn_input": 600, "dnn_conf": 0.25,
                "haar_scale": 1.08, "haar_neighbors": 4, "haar_min_size": (25, 25),
                "iou_thresh": 0.5,
            },
        }
        
        p = params.get(sensitivity, params["medium"])
        
        # DNN 检测
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
        
        # Haar 补充检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        detected = self.haar_cascade.detectMultiScale(
            gray, scaleFactor=p["haar_scale"], minNeighbors=p["haar_neighbors"],
            minSize=p["haar_min_size"], maxSize=(int(w * 0.4), int(h * 0.4))
        )
        
        for (fx, fy, fw, fh) in detected:
            if self._is_valid_face(int(fx), int(fy), int(fw), int(fh), h, w):
                all_faces.append(((int(fx), int(fy), int(fw), int(fh)), 0.35))
        
        # NMS 去重
        unique_faces = self._nms(all_faces, iou_threshold=p["iou_thresh"])
        
        # 扩展边界框
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
                      whitelist_features: List[np.ndarray] = None,
                      sensitivity: str = "medium") -> np.ndarray:
        """
        处理单张图片，对非白名单人脸应用隐私保护效果
        
        Args:
            image: 输入图片
            effect: 效果类型
            whitelist_features: 白名单人员的人脸特征向量列表
            sensitivity: 检测灵敏度
        """
        faces = self.detect_faces(image, sensitivity)
        
        if not faces:
            return image
        
        result = image.copy()
        
        for face in faces:
            x, y, w, h = face
            
            # 白名单特征比对
            if whitelist_features and self.face_recognizer is not None:
                # 提取当前人脸特征
                current_feature = self.extract_face_feature(image, face)
                
                if current_feature is not None:
                    # 与每个白名单特征比对
                    is_whitelisted = False
                    for wl_feature in whitelist_features:
                        similarity = self.compute_similarity(current_feature, wl_feature)
                        if similarity > self.WHITELIST_THRESHOLD:
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


def load_known_faces(known_faces_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    加载白名单人脸特征
    
    Args:
        known_faces_dir: 白名单人脸图片目录
        
    Returns:
        (人脸图片列表, 人脸特征向量列表)
    """
    known_faces = []
    face_features = []
    
    processor = FaceProcessor()
    known_dir = Path(known_faces_dir)
    
    if not known_dir.exists():
        return known_faces, face_features
    
    for img_path in known_dir.glob("*"):
        if img_path.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
            continue
        
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        
        faces = processor.detect_faces(img, "medium")
        if not faces:
            continue
        
        # 提取人脸特征（取最大的人脸框）
        best_face = max(faces, key=lambda f: f[2] * f[3])
        feature = processor.extract_face_feature(img, best_face)
        
        if feature is not None:
            known_faces.append(img)
            face_features.append(feature)
            print(f"✓ 白名单加载: {img_path.name}, 特征维度 {feature.shape}")
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
    print("人脸处理器初始化完成")
