#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
幸福的照片隐私打码工具 - Web 界面
Flask 应用主程序
"""

import os
import sys
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import uuid
import base64
from io import BytesIO
import time

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from face_processor import FaceProcessor, load_known_faces

app = Flask(__name__)
app.secret_key = 'face-privacy-tool-secret-2024'

# 配置
UPLOAD_FOLDER = Path(__file__).parent / 'static' / 'uploads'
RESULT_FOLDER = Path(__file__).parent / 'static' / 'results'
KNOWN_FACES_FOLDER = Path(__file__).parent / 'known_faces'
MODELS_FOLDER = Path(__file__).parent / 'models'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

# 创建必要目录
for folder in [UPLOAD_FOLDER, RESULT_FOLDER, KNOWN_FACES_FOLDER, MODELS_FOLDER]:
    folder.mkdir(parents=True, exist_ok=True)

# 初始化人脸处理器
processor = None

def get_processor():
    """获取或创建人脸处理器实例"""
    global processor
    if processor is None:
        processor = FaceProcessor(str(MODELS_FOLDER))
    return processor

def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def image_to_base64(image, format='.jpg'):
    """将 OpenCV 图片转换为 base64 字符串"""
    if format == '.png':
        encode_param = [cv2.IMWRITE_PNG_COMPRESSION, 9]
        _, buffer = cv2.imencode('.png', image, encode_param)
    else:
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, 95]
        _, buffer = cv2.imencode('.jpg', image, encode_param)
    
    return base64.b64encode(buffer).decode('utf-8')

def base64_to_image(base64_str):
    """将 base64 字符串转换为 OpenCV 图片"""
    # 移除 data:image/xxx;base64, 前缀
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]
    
    buffer = base64.b64decode(base64_str)
    np_arr = np.frombuffer(buffer, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.route('/')
def index():
    """主页"""
    # 获取已注册的白名单人员
    whitelist_names = []
    for f in KNOWN_FACES_FOLDER.glob('*'):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            whitelist_names.append(f.stem)
    
    return render_template('index.html', whitelist=whitelist_names)

@app.route('/upload', methods=['POST'])
def upload_file():
    """上传并处理单张图片"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': '不支持的文件格式'}), 400
    
    # 读取图片
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': '无法解析图片'}), 400
    
    # 获取效果参数
    effect = request.form.get('effect', 'blur')
    
    # 加载白名单
    _, whitelist_boxes = load_known_faces(str(KNOWN_FACES_FOLDER))
    
    # 处理图片
    proc = get_processor()
    result = proc.process_image(image, effect, whitelist_boxes)
    
    # 转换为 base64
    original_b64 = image_to_base64(image)
    result_b64 = image_to_base64(result)
    
    # 检测人脸数量
    faces = proc.detect_faces(image)
    face_count = len(faces)
    
    return jsonify({
        'original': f'data:image/jpeg;base64,{original_b64}',
        'result': f'data:image/jpeg;base64,{result_b64}',
        'face_count': face_count,
        'whitelist_count': len(whitelist_boxes)
    })

@app.route('/upload_base64', methods=['POST'])
def upload_base64():
    """接收 base64 编码的图片并处理"""
    data = request.get_json()
    
    if not data or 'image' not in data:
        return jsonify({'error': '没有收到图片数据'}), 400
    
    try:
        image = base64_to_image(data['image'])
        if image is None:
            return jsonify({'error': '无法解析图片'}), 400
    except Exception as e:
        return jsonify({'error': f'图片解析失败: {str(e)}'}), 400
    
    effect = data.get('effect', 'blur')
    
    # 加载白名单
    _, whitelist_boxes = load_known_faces(str(KNOWN_FACES_FOLDER))
    
    # 处理图片
    proc = get_processor()
    result = proc.process_image(image, effect, whitelist_boxes)
    
    # 转换为 base64
    result_b64 = image_to_base64(result)
    
    # 检测人脸
    faces = proc.detect_faces(image)
    face_count = len(faces)
    
    return jsonify({
        'result': f'data:image/jpeg;base64,{result_b64}',
        'face_count': face_count,
        'whitelist_count': len(whitelist_boxes)
    })

@app.route('/batch', methods=['POST'])
def batch_process():
    """批量处理多张图片"""
    if 'files' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': '没有选择文件'}), 400
    
    effect = request.form.get('effect', 'blur')
    proc = get_processor()
    
    # 加载白名单
    _, whitelist_boxes = load_known_faces(str(KNOWN_FACES_FOLDER))
    
    results = []
    session_id = str(uuid.uuid4())[:8]
    
    for i, file in enumerate(files):
        if file.filename == '' or not allowed_file(file.filename):
            continue
        
        try:
            file_bytes = np.frombuffer(file.read(), np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            if image is None:
                continue
            
            result = proc.process_image(image, effect, whitelist_boxes)
            
            # 保存结果
            filename = secure_filename(file.filename)
            result_path = RESULT_FOLDER / f"{session_id}_{i}_{filename}"
            
            if result_path.suffix.lower() in ['.jpg', '.jpeg']:
                cv2.imwrite(str(result_path), result, [cv2.IMWRITE_JPEG_QUALITY, 95])
            else:
                cv2.imwrite(str(result_path), result)
            
            faces = proc.detect_faces(image)
            
            results.append({
                'original_name': filename,
                'result_name': result_path.name,
                'face_count': len(faces)
            })
            
        except Exception as e:
            print(f"处理文件失败: {file.filename}, 错误: {e}")
            continue
    
    return jsonify({
        'session_id': session_id,
        'results': results,
        'total': len(results)
    })

@app.route('/whitelist/add', methods=['POST'])
def add_whitelist():
    """添加白名单人员"""
    if 'file' not in request.files or 'name' not in request.form:
        return jsonify({'error': '缺少文件或姓名'}), 400
    
    file = request.files['file']
    name = request.form['name'].strip()
    
    if not name:
        return jsonify({'error': '姓名不能为空'}), 400
    
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({'error': '无效的文件'}), 400
    
    # 检测是否包含人脸
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': '无法解析图片'}), 400
    
    proc = get_processor()
    faces = proc.detect_faces(image)
    
    if not faces:
        return jsonify({'error': '未检测到人脸，请上传包含人脸的清晰照片'}), 400
    
    # 保存到白名单目录
    safe_name = secure_filename(name)
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    save_path = KNOWN_FACES_FOLDER / f"{safe_name}{ext}"
    
    # 避免覆盖
    counter = 1
    while save_path.exists():
        save_path = KNOWN_FACES_FOLDER / f"{safe_name}_{counter}{ext}"
        counter += 1
    
    cv2.imwrite(str(save_path), image)
    
    return jsonify({
        'success': True,
        'name': save_path.stem,
        'face_count': len(faces)
    })

@app.route('/whitelist/list')
def list_whitelist():
    """获取白名单列表"""
    whitelist = []
    for f in KNOWN_FACES_FOLDER.glob('*'):
        if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            # 读取图片获取人脸数量
            img = cv2.imread(str(f))
            if img is not None:
                proc = get_processor()
                faces = proc.detect_faces(img)
                whitelist.append({
                    'name': f.stem,
                    'filename': f.name,
                    'face_count': len(faces)
                })
    
    return jsonify({'whitelist': whitelist})

@app.route('/whitelist/delete/<name>')
def delete_whitelist(name):
    """删除白名单人员"""
    safe_name = secure_filename(name)
    
    for f in KNOWN_FACES_FOLDER.glob('*'):
        if f.stem == safe_name and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
            f.unlink()
            return jsonify({'success': True})
    
    return jsonify({'error': '未找到该人员'}), 404

@app.route('/download/<filename>')
def download_result(filename):
    """下载处理结果"""
    safe_filename = secure_filename(filename)
    file_path = RESULT_FOLDER / safe_filename
    
    if not file_path.exists():
        return jsonify({'error': '文件不存在'}), 404
    
    return send_file(file_path, as_attachment=True)

@app.route('/download_batch/<session_id>')
def download_batch(session_id):
    """下载批量处理结果（ZIP）"""
    import zipfile
    from tempfile import NamedTemporaryFile
    
    # 查找该批次的所有文件
    batch_files = list(RESULT_FOLDER.glob(f"{session_id}_*"))
    
    if not batch_files:
        return jsonify({'error': '没有找到该批次的文件'}), 404
    
    # 创建临时 ZIP 文件
    with NamedTemporaryFile(suffix='.zip', delete=False) as tmp:
        with zipfile.ZipFile(tmp.name, 'w', zipfile.ZIP_DEFLATED) as zf:
            for f in batch_files:
                zf.write(f, f.name.replace(f"{session_id}_", ""))
        
        return send_file(tmp.name, as_attachment=True, download_name=f'隐私打码结果_{session_id}.zip')

@app.route('/detect', methods=['POST'])
def detect_faces():
    """仅检测人脸，不做处理"""
    if 'file' not in request.files:
        return jsonify({'error': '没有上传文件'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': '没有选择文件'}), 400
    
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None:
        return jsonify({'error': '无法解析图片'}), 400
    
    proc = get_processor()
    faces = proc.detect_faces(image)
    
    # 在图片上绘制人脸框
    result = image.copy()
    for (x, y, w, h) in faces:
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    result_b64 = image_to_base64(result)
    
    return jsonify({
        'image': f'data:image/jpeg;base64,{result_b64}',
        'face_count': len(faces),
        'faces': [{'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)} for (x, y, w, h) in faces]
    })

@app.route('/health')
def health():
    """健康检查"""
    proc = get_processor()
    whitelist_count = len(list(KNOWN_FACES_FOLDER.glob('*')))
    
    return jsonify({
        'status': 'ok',
        'processor': 'loaded' if proc else 'error',
        'whitelist_count': whitelist_count
    })

if __name__ == '__main__':
    print("=" * 50)
    print("幸福的照片隐私打码工具")
    print("=" * 50)
    print("启动 Web 服务器...")
    print(f"上传目录: {UPLOAD_FOLDER}")
    print(f"结果目录: {RESULT_FOLDER}")
    print(f"白名单目录: {KNOWN_FACES_FOLDER}")
    print()
    
    # 初始化处理器
    get_processor()
    
    # 启动 Flask
    app.run(host='0.0.0.0', port=5000, debug=True)
