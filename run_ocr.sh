#!/bin/bash
# PDF OCR 批量处理 - 启动脚本
# 自动检测环境并运行 OCR 处理

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OCR_SCRIPT="${SCRIPT_DIR}/pdf_ocr_stable.py"

# 默认配置
DEFAULT_OUTPUT_DIR="${SCRIPT_DIR}/ocr_output"
LOG_FILE="${SCRIPT_DIR}/ocr_run.log"

echo "========================================"
echo "  PDF OCR 批量处理工具"
echo "  版本：2.0.0-stable"
echo "========================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}错误：未找到 Python3${NC}"
    exit 1
fi

echo "✓ Python: $(python3 --version)"

# 检查 Conda 环境
if [ -n "$CONDA_PREFIX" ]; then
    echo "✓ Conda 环境：$CONDA_PREFIX"
else
    echo -e "${YELLOW}提示：未激活 Conda 环境${NC}"
    echo "  建议运行：conda activate ocr_gpu_clean"
    echo ""
fi

# 检查脚本
if [ ! -f "$OCR_SCRIPT" ]; then
    echo -e "${RED}错误：未找到 OCR 脚本：$OCR_SCRIPT${NC}"
    exit 1
fi

# 显示帮助
show_help() {
    echo ""
    echo "用法：$0 [选项] <PDF 文件或目录>"
    echo ""
    echo "选项:"
    echo "  -h, --help          显示帮助信息"
    echo "  -o, --output DIR    输出目录（默认：./ocr_output）"
    echo "  -g, --gpu MEM       GPU 显存限制（GB，默认：自动）"
    echo "  -w, --workers NUM   并行线程数（默认：自动）"
    echo "  -d, --debug         调试模式"
    echo "  -b, --background    后台运行"
    echo ""
    echo "示例:"
    echo "  $0 /path/to/pdfs"
    echo "  $0 -o /output/dir /path/to/pdfs"
    echo "  $0 -g 8 -w 2 /path/to/pdfs"
    echo "  $0 -b /path/to/pdfs  # 后台运行"
    echo ""
}

# 解析参数
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
GPU_MEMORY=""
WORKERS=""
DEBUG=""
BACKGROUND=""
INPUT_PATH=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            exit 0
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -g|--gpu)
            GPU_MEMORY="--gpu-memory $2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="--workers $2"
            shift 2
            ;;
        -d|--debug)
            DEBUG="--debug"
            shift
            ;;
        -b|--background)
            BACKGROUND="background"
            shift
            ;;
        *)
            if [ -z "$INPUT_PATH" ]; then
                INPUT_PATH="$1"
            else
                echo -e "${RED}错误：未知参数：$1${NC}"
                show_help
                exit 1
            fi
            shift
            ;;
    esac
done

# 检查输入路径
if [ -z "$INPUT_PATH" ]; then
    echo -e "${RED}错误：请指定 PDF 文件或目录路径${NC}"
    show_help
    exit 1
fi

if [ ! -e "$INPUT_PATH" ]; then
    echo -e "${RED}错误：路径不存在：$INPUT_PATH${NC}"
    exit 1
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 构建命令
CMD="python3 $OCR_SCRIPT $INPUT_PATH -o $OUTPUT_DIR $GPU_MEMORY $WORKERS $DEBUG"

echo ""
echo "========================================"
echo "  开始处理"
echo "========================================"
echo "输入：$INPUT_PATH"
echo "输出：$OUTPUT_DIR"
echo "命令：$CMD"
echo ""

# 运行
if [ "$BACKGROUND" == "background" ]; then
    echo "后台运行中..."
    nohup $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "进程 ID: $PID"
    echo "日志文件：$LOG_FILE"
    echo ""
    echo "监控命令:"
    echo "  查看日志：tail -f $LOG_FILE"
    echo "  查看状态：cat ${SCRIPT_DIR}/ocr_status.txt"
    echo "  查看进程：ps aux | grep $PID"
else
    $CMD
fi

echo ""
echo "========================================"
echo "  完成"
echo "========================================"
