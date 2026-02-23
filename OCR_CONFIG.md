# PDF OCR 批量处理工具 - 配置说明

## 快速开始

### 1. 环境准备

```bash
# 激活 Conda 环境（如果有）
conda activate ocr_gpu_clean

# 或安装依赖
pip install paddlepaddle-gpu paddleocr PyMuPDF opencv-python
```

### 2. 运行方式

#### 方式一：使用启动脚本（推荐）

```bash
# 基本用法
./run_ocr.sh /path/to/pdfs

# 指定输出目录
./run_ocr.sh -o /output/dir /path/to/pdfs

# 自定义 GPU 显存和线程数
./run_ocr.sh -g 8 -w 2 /path/to/pdfs

# 后台运行
./run_ocr.sh -b /path/to/pdfs

# 调试模式
./run_ocr.sh -d /path/to/pdfs
```

#### 方式二：直接运行 Python 脚本

```bash
# 基本用法
python3 pdf_ocr_stable.py /path/to/pdfs

# 自定义配置
python3 pdf_ocr_stable.py /path/to/pdfs -o /output --gpu-memory 8 --workers 2

# 查看帮助
python3 pdf_ocr_stable.py --help
```

## 配置说明

### 自动配置（推荐）

脚本会自动根据硬件配置选择最优参数：

| GPU 显存 | 推荐配置 | 适用场景 |
|----------|----------|----------|
| ≥24GB | 16GB 显存，6 线程 | 高性能服务器 |
| ≥16GB | 9GB 显存，3 线程 | RTX 3080/3090 |
| ≥12GB | 7GB 显存，2 线程 | RTX 3060/4070 |
| <12GB | 5GB 显存，2 线程 | 入门级 GPU |

### 手动配置参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--gpu-memory` | GPU 显存限制（GB） | 自动 |
| `--workers` | 并行工作线程数 | 自动 |
| `-o` | 输出目录 | ./ocr_output |
| `--debug` | 调试模式 | 关闭 |

## 硬件适配

### NVIDIA GPU

```bash
# RTX 3080 Ti (16GB)
./run_ocr.sh -g 9 -w 3 /path/to/pdfs

# RTX 3090 (24GB)
./run_ocr.sh -g 16 -w 6 /path/to/pdfs

# RTX 3060 (12GB)
./run_ocr.sh -g 7 -w 2 /path/to/pdfs
```

### CPU 模式（无 GPU）

```bash
# 自动使用 CPU 模式
python3 pdf_ocr_stable.py /path/to/pdfs
```

## 监控与维护

### 查看进度

```bash
# 查看实时日志
tail -f ocr_process.log

# 查看当前状态
cat ocr_status.txt

# 查看已完成文件
ls -lht ocr_output/
```

### 后台运行监控

```bash
# 启动后台任务
./run_ocr.sh -b /path/to/pdfs

# 查看日志
tail -f ocr_run.log

# 查看进程
ps aux | grep pdf_ocr

# 停止任务
pkill -f pdf_ocr_stable
```

## 故障排除

### 显存不足错误

```
错误：CUDA out of memory
```

**解决方案：**
1. 降低 GPU 显存限制：`-g 6`
2. 减少并行线程：`-w 2`
3. 降低图片 DPI：修改脚本中的 `dpi` 参数

### OCR 识别错误

```
错误：Tensor holds no memory
```

**解决方案：**
1. 脚本已自动处理（重试机制）
2. 如持续出现，减少并行线程数

### 依赖问题

```
错误：No module named 'paddleocr'
```

**解决方案：**
```bash
# 重新安装依赖
pip install --upgrade paddlepaddle-gpu paddleocr
```

## 性能优化

### 处理速度对比

| 配置 | 速度 | 错误率 | 适用场景 |
|------|------|--------|----------|
| 3 线程 + 每页清理 | 45 页/分 | 0% | 稳定生产 |
| 4 线程 + 每页清理 | 55 页/分 | <1% | 日常使用 |
| 6 线程 + 每页清理 | 70 页/分 | <2% | 高性能需求 |

### 推荐配置

**稳定型（0 错误率）：**
```bash
./run_ocr.sh -g 9 -w 3 /path/to/pdfs
```

**平衡型（<1% 错误率）：**
```bash
./run_ocr.sh -g 10 -w 4 /path/to/pdfs
```

**性能型（<2% 错误率）：**
```bash
./run_ocr.sh -g 12 -w 6 /path/to/pdfs
```

## 输出格式

每个 PDF 生成一个 TXT 文件：

```
=== 第 1 页 ===
识别内容...

=== 第 2 页 ===
识别内容...
```

## 状态文件

`ocr_status.txt` 包含当前处理状态：

```
完成时间：2024 年 01 月 01 日 12:00:00
当前文件：示例文件
输出路径：./ocr_output/示例文件_ocr.txt
文件大小：1234.5 KB
进度：1/24
剩余：23 个文件
```

## 桌面通知

每完成一个文件会发送桌面通知（需要桌面环境支持）。

## 许可证

本脚本仅供学习和研究使用。
