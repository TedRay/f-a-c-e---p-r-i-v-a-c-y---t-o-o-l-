# 幸福的照片隐私打码工具

> 广州十方缘森林疗愈活动照片批量隐私保护处理工具

## 功能特点

- 🔒 **隐私保护**: 自动检测人脸并应用隐私打码效果
- 👥 **白名单机制**: 指定人员人脸保持原样，不做处理
- 📦 **批量处理**: 支持一次性处理多张照片
- 🎨 **多种效果**: 模糊平滑、像素化、表情包贴纸

## 安装

```bash
# 克隆仓库
git clone <repo-url>
cd face-privacy-tool

# 安装依赖
pip3 install -r requirements.txt

# 启动服务
bash start.sh
# 或
python3 app.py
```

## 使用方法

1. 启动服务后访问 `http://localhost:5000`
2. 在「白名单管理」中添加需要保留人脸的人员照片
3. 在「单张处理」或「批量处理」中上传照片
4. 选择打码效果并点击处理
5. 下载处理后的照片

## 打码效果说明

| 效果 | 说明 |
|------|------|
| 🌫️ 模糊平滑 | 使用双边滤波+高斯模糊，柔和处理人脸 |
| 🟩 像素化 | 马赛克效果，清晰遮挡人脸 |
| 😊 表情包 | 可爱的黄色笑脸遮挡人脸 |

## 目录结构

```
face-privacy-tool/
├── app.py              # Flask 主程序
├── face_processor.py   # 人脸检测与处理核心
├── templates/
│   └── index.html      # Web 界面
├── static/
│   ├── uploads/        # 上传文件临时目录
│   └── results/        # 处理结果目录
├── known_faces/        # 白名单人脸照片
├── models/             # 人脸检测模型
├── requirements.txt    # Python 依赖
├── start.sh            # 启动脚本
└── README.md           # 本文件
```

## 技术栈

- **后端**: Flask + OpenCV (DNN 人脸检测)
- **前端**: 原生 HTML/CSS/JS
- **无需 dlib**: 使用 OpenCV 内置 DNN 模型，安装简单

## 注意事项

- 首次运行会自动下载人脸检测模型 (~5MB)
- 建议上传清晰、正面的人脸照片以获得最佳检测效果
- 白名单照片请使用单人清晰正面照

## 许可证

MIT License

---

**广州十方缘森林疗愈** · 保护每一份隐私
