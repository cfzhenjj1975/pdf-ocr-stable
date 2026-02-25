# PDF OCR 批量处理系统 - 专业版

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PaddleOCR](https://img.shields.io/badge/PaddleOCR-v2.7+-green.svg)](https://github.com/PaddlePaddle/PaddleOCR)
[![License](https://img.shields.io/badge/license-Apache%202.0-red.svg)](LICENSE)

高精度 PDF 批量 OCR 识别系统，支持横排/竖排/表格/分栏/照片智能识别，0 错误率前提下最快速度。

## 🚀 特性

- **智能版面分析**: 横排/竖排/表格/分栏/照片自动识别
- **多模型切换**: 简体/繁体中文模型自动选择
- **后处理校对**: 错别字纠正/标点规范化/繁简转换
- **页码一致性**: 确保输出页码与源文件完全一致
- **DOCX 输出**: 带版式和插图的高质量输出
- **实时监控**: CPU/GPU/内存/进度实时面板

## 📊 性能指标

| 指标 | v12.0 (PaddleOCR) | v13.0 (混合架构) |
|------|------------------|-----------------|
| **速度** | 45-50 页/分钟 | 80-100 页/分钟 |
| **置信度** | 0.85-0.90 | 0.88-0.92 |
| **显存占用** | ~3GB | 8-10GB |
| **DPI** | 185 | 185 |

## 🔧 安装

### 环境要求

- Python 3.9+
- GPU: NVIDIA (≥4GB 显存，推荐 16GB+)
- CUDA 11.8+
- 系统内存：≥16GB (推荐 32GB+)

### 快速安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/pdf-ocr-pro.git
cd pdf-ocr-pro

# 创建 Conda 环境
conda create -n ocr_gpu python=3.9
conda activate ocr_gpu

# 安装依赖
pip install -r requirements.txt

# 安装 PaddleOCR (GPU 版)
pip install paddlepaddle-gpu==2.6.0
pip install paddleocr==2.7.3
```

### 完整依赖

```bash
pip install -r requirements.txt
```

## 📖 使用方法

### 快速启动

```bash
# v12.0 PaddleOCR 版 (稳定)
bash start_v12_final.sh /path/to/pdf/folder

# v13.0 混合架构版 (推荐)
bash start_v13_hybrid.sh /path/to/pdf/folder
```

### 命令行参数

```bash
python pdf_ocr_v12_final.py /path/to/pdfs -o /path/to/output
```

### 输出目录

```
/media/zjj/leidian/leidian/ocr_output_v12_final/
├── *.docx              # OCR 识别结果
└── ocr_status.txt      # 实时进度状态
```

## 📁 版本说明

| 版本 | 说明 | 状态 |
|------|------|------|
| **v12.0** | PaddleOCR 最终锁定版 (DPI=185) | ✅ 稳定 |
| **v13.0** | 混合架构 (PaddleOCR + Qwen2.5-VL) | ⭐ 推荐 |

### v12.0 配置参数

```python
DPI = 185
DET_DB_THRESH = 0.4
DET_DB_BOX_THRESH = 0.42
DET_DB_UNCLIP_RATIO = 1.3
DROP_SCORE = 0.55
```

### v13.0 混合架构

- **简单页面** (横排/表格): PaddleOCR (快)
- **复杂页面** (竖排/古籍): Qwen2.5-VL (准)

## 📊 监控工具

### 实时监控面板

```bash
python ocr_monitor_v2.py
```

### 快速状态查看

```bash
bash ocr_quick_status.sh
```

## 📝 参数优化

详细参数调整指南见：[PARAMS_GUIDE.md](PARAMS_GUIDE.md)

### 快速调整

```python
# 编辑配置类 (pdf_ocr_v12_final.py 第 38-60 行)
class Config:
    DPI = 185                    # 清晰度 vs 速度
    DET_DB_THRESH = 0.4          # 文字检测阈值
    DET_DB_BOX_THRESH = 0.42     # 检测框阈值
    DET_DB_UNCLIP_RATIO = 1.3    # 检测框扩展
    DROP_SCORE = 0.55            # 置信度过滤
```

## 📋 测试报告

- [参数优化测试](test_dpi185_output.log)
- [效果检查报告](EFFECT_CHECK_REPORT.md)
- [vLLM 评估报告](VLLM_QWEN_EVALUATION.md)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

Apache License 2.0

## 📧 联系

- Email: ocr@local.dev
- GitHub: https://github.com/yourusername/pdf-ocr-pro

---

**最后更新**: 2026-02-25  
**版本**: v12.0 / v13.0
