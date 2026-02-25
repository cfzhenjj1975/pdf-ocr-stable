#!/bin/bash
# PDF OCR 专业版 v12.0 最终锁定版 - 启动脚本
# DPI=185 + 参数优化测试最优值

export LD_LIBRARY_PATH=/usr/local/cudnn-linux-x86_64-8.9.7.29_cuda11-archive/lib:$LD_LIBRARY_PATH
source /home/zjj/miniconda3/etc/profile.d/conda.sh
conda activate ocr_gpu_clean
export PYTHONUNBUFFERED=1

cd /home/zjj/pdf-ocr-stable

echo "=============================================="
echo "PDF OCR 专业版 v12.0 最终锁定版"
echo "=============================================="
echo "配置：DPI=185 + 参数优化测试最优值"
echo "  ✓ DPI: 185"
echo "  ✓ DET_THRESH: 0.4"
echo "  ✓ BOX_THRESH: 0.42"
echo "  ✓ UNCLIP_RATIO: 1.3"
echo "  ✓ DROP_SCORE: 0.55"
echo "=============================================="

python3 -u pdf_ocr_v12_final.py /media/zjj/leidian/leidian \
    -o /media/zjj/leidian/leidian/ocr_output_v12_final \
    "$@"
