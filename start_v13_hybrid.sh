#!/bin/bash
# PDF OCR 专业版 v13.0 混合架构版 - 启动脚本
# PaddleOCR (快) + Qwen2.5-VL (准)

export LD_LIBRARY_PATH=/home/zjj/miniconda3/envs/ocr_gpu_clean/lib:$LD_LIBRARY_PATH
source /home/zjj/miniconda3/etc/profile.d/conda.sh
conda activate ocr_gpu_clean
export PYTHONUNBUFFERED=1

cd /home/zjj/pdf-ocr-stable

echo "=============================================="
echo "PDF OCR 专业版 v13.0 混合架构版"
echo "=============================================="
echo "架构：PaddleOCR (快) + Qwen2.5-VL (准)"
echo "  ✓ PaddleOCR: 横排/表格 (45-50 页/分)"
echo "  ✓ Qwen2.5-VL: 竖排/古籍 (更高质量)"
echo "  ✓ 预估速度：80-100 页/分钟"
echo "=============================================="

# 检查 vLLM 服务
echo ""
echo "检查 vLLM 服务状态..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ vLLM 服务运行中"
else
    echo "⚠️  vLLM 服务未启动，将仅使用 PaddleOCR"
    echo "启动 vLLM 服务命令:"
    echo "  python -m vllm.entrypoints.api_server --model Qwen/Qwen2.5-VL-7B-Instruct --port 8000"
fi
echo ""

python3 -u pdf_ocr_v13_hybrid.py /media/zjj/leidian/leidian \
    -o /media/zjj/leidian/leidian/ocr_output_v13_hybrid \
    "$@"
