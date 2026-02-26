#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PDF 批量 OCR 处理脚本 - v4.0 智能识别版
修复：
1. 首先自动识别文档类型（古籍/现代）
2. 古籍：从右到左、从上到下，严格框线版面
3. 现代：从左到右、从上到下，标准 Markdown
4. 质量报告分离
"""

import os
import sys
import gc
import time
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import multiprocessing

warnings.filterwarnings("ignore")
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

# ==================== 版本信息 ====================
VERSION = "v4.0-SmartDoc"
VERSION_NOTE = "智能识别 + 古籍专用版面"
TARGET_SPEED = 120  # 页/分钟

# ==================== 核心配置 ====================

INPUT_DIR = "/media/zjj/leidian1/leidian"
OUTPUT_DIR = "/media/zjj/leidian1/leidian/ocr_output_v4_smart"
REPORT_DIR = "/media/zjj/leidian1/leidian/ocr_quality_reports_v4"

# OCR 配置（全模块 TensorRT FP16 加速）
OCR_CONFIG = {
    "use_hpip": True,
    "device": "gpu:0",
    "pipeline": "OCR",
    "text_recognition_batch_size": 512,  # 增大 batch 到 512
    # TensorRT 加速配置（所有模块启用 FP16）
    "trt_config": {
        "precision_mode": "FP16",  # FP16 半精度加速
        "trt_use_dynamic_shapes": True,
        "trt_min_shape": [1, 3, 32, 32],
        "trt_opt_shape": [1, 3, 48, 320],
        "trt_max_shape": [8, 3, 48, 3200],
        "trt_static_cache": True,  # 启用静态缓存
        "trt_workspace_size": 1024,  # 1GB 工作空间
    }
}

# 性能配置（CPU 供数优化版 - 目标 120 页/分钟）
PERF_CONFIG = {
    "max_workers": 12,           # OCR 并行 12 线程
    "dpi": 190,                  # 锁定 DPI 190
    "image_max_size": 1200,      # 降低图片尺寸到 1200
    "prefetch_pages": 200,       # CPU 预取 200 页（翻倍）
    "cpu_workers": 16,           # 16 线程解码（+4）
    "cpu_decode_prefetch": 100,  # 预解码 100 页（翻倍）
}

# 古籍识别配置
ANCIENT_DETECT_CONFIG = {
    "keywords_ancient": [
        "撰", "撰并", "序", "跋", "卷", "纪", "传", "志", "表", "谱",
        "皇", "帝", "诏", "谕", "奏", "疏", "古文", "文言文",
        "光绪", "乾隆", "嘉庆", "道光", "咸丰", "同治", "宣统",
        "康熙", "雍正", "明朝", "清朝", "宋朝", "唐朝", "元年",
        "岁次", "干支", "甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸",
        "子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"
    ],
    "keywords_modern": [
        "报告", "分析", "研究", "调查", "统计", "数据", "技术", "工程",
        "科学", "技术", "公司", "企业", "单位", "部门", "年", "月", "日",
        "摘要", "关键词", "引言", "结论", "参考", "文献", "图表", "附录"
    ],
    "confidence_threshold": 0.70,  # 置信度阈值
}

STATE_FILE = "/media/zjj/leidian1/leidian/.paddlex_ocr_v4_state.json"

# ====================================================


def clear_gpu_memory():
    """清理 GPU 显存"""
    gc.collect()
    import paddle
    if paddle.is_compiled_with_cuda():
        paddle.device.cuda.empty_cache()
        paddle.device.cuda.synchronize()
        print(f"  ✓ GPU 显存已清理")


def check_and_clear_gpu_memory(threshold_mb=14000):
    """检查并清理 GPU 显存（阈值 14GB）"""
    import paddle
    if paddle.is_compiled_with_cuda():
        mem_info = paddle.device.cuda.memory_allocated()
        mem_mb = mem_info / 1024 / 1024
        if mem_mb > threshold_mb:
            print(f"  ⚠️  显存占用 {mem_mb:.0f}MB > {threshold_mb}MB，自动清理...")
            clear_gpu_memory()
        return mem_mb
    return 0


class PageCache:
    """页面缓存管理器"""
    
    def __init__(self, max_pages=500):
        self.max_pages = max_pages
        self.cache = {}
        self.access_order = []
        
    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None
    
    def put(self, key, value):
        if len(self.cache) >= self.max_pages:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]
        self.cache[key] = value
        self.access_order.append(key)
    
    def clear(self):
        self.cache.clear()
        self.access_order.clear()


def detect_document_type(pdf_name: str, first_page_text: str) -> str:
    """
    智能识别文档类型
    返回："ancient"（古籍）或 "modern"（现代）
    """
    ancient_score = 0
    modern_score = 0
    
    # 文件名检测
    pdf_name_lower = pdf_name.lower()
    for keyword in ANCIENT_DETECT_CONFIG["keywords_ancient"]:
        if keyword in pdf_name_lower:
            ancient_score += 2
    
    for keyword in ANCIENT_DETECT_CONFIG["keywords_modern"]:
        if keyword in pdf_name_lower:
            modern_score += 2
    
    # 文本内容检测
    text_lower = first_page_text.lower()
    for keyword in ANCIENT_DETECT_CONFIG["keywords_ancient"]:
        if keyword in text_lower:
            ancient_score += 1
    
    for keyword in ANCIENT_DETECT_CONFIG["keywords_modern"]:
        if keyword in text_lower:
            modern_score += 1
    
    # 判断结果
    if ancient_score > modern_score * 1.5:
        return "ancient"
    elif modern_score > ancient_score * 1.5:
        return "modern"
    else:
        # 分数接近，使用更多规则
        # 古籍特征：短文本、繁体字、竖排
        if len(first_page_text) < 200 and any(c in first_page_text for c in "繁體字"):
            return "ancient"
        # 默认按现代文档处理
        return "modern"


def init_ocr_pipeline():
    """初始化 PaddleOCR 流水线（全模块 TensorRT FP16 加速）"""
    from paddlex import create_pipeline

    print(f"  ╔════════════════════════════════════════════════════════╗")
    print(f"  ║  OCR v4.0 智能识别版 {VERSION:16s}                      ║")
    print(f"  ║  调整：{VERSION_NOTE:44s}  ║")
    print(f"  ║  目标：{TARGET_SPEED} 页/分钟                                     ║")
    print(f"  ╚════════════════════════════════════════════════════════╝")
    print(f"")
    print(f"  【核心功能】")
    print(f"    1. 智能识别：自动区分古籍/现代文档")
    print(f"    2. 古籍版面：从右到左、从上到下、框线分隔")
    print(f"    3. 现代版面：标准 Markdown 格式")
    print(f"    4. 质量报告：单独 JSON 文件")
    print(f"")
    print(f"  【全模块 TensorRT FP16 加速】")
    print(f"    - UVDoc: FP16 TensorRT ✓")
    print(f"    - 文本行方向：FP16 TensorRT ✓")
    print(f"    - 文本检测：FP16 TensorRT ✓")
    print(f"    - 文本识别：FP16 TensorRT ✓")
    print(f"    - 批处理：{OCR_CONFIG['text_recognition_batch_size']}")
    print(f"")
    print(f"  【GPU 资源优化】")
    print(f"    - TensorRT 工作空间：1GB")
    print(f"    - 动态形状：启用")
    print(f"    - 静态缓存：启用")
    print(f"")
    print(f"  【模型配置】")
    print(f"    - Pipeline: {OCR_CONFIG['pipeline']}")
    print(f"    - DPI: {PERF_CONFIG['dpi']}")
    print(f"    - 图片尺寸：{PERF_CONFIG['image_max_size']}")
    print(f"")
    print(f"  【输出格式】")
    print(f"    - 古籍：框线版面 Markdown")
    print(f"    - 现代：标准 Markdown")
    print(f"    - 质量报告：单独 JSON 文件")
    print(f"")

    print(f"  正在初始化 PaddleOCR 流水线（全模块 TensorRT FP16）...")
    
    # 创建 PaddleX 流水线，应用 TensorRT 配置到所有模块
    pipeline = create_pipeline(
        pipeline=OCR_CONFIG["pipeline"],
        use_hpip=OCR_CONFIG["use_hpip"],
        device=OCR_CONFIG["device"],
        # 全局 TensorRT 配置（应用到所有模块）
        trt_precision=OCR_CONFIG["trt_config"]["precision_mode"],
        trt_use_dynamic_shapes=OCR_CONFIG["trt_config"]["trt_use_dynamic_shapes"],
        trt_min_shape=OCR_CONFIG["trt_config"]["trt_min_shape"],
        trt_opt_shape=OCR_CONFIG["trt_config"]["trt_opt_shape"],
        trt_max_shape=OCR_CONFIG["trt_config"]["trt_max_shape"],
        trt_static_cache=OCR_CONFIG["trt_config"]["trt_static_cache"],
        trt_workspace_size=OCR_CONFIG["trt_config"]["trt_workspace_size"],
    )
    print(f"  ✓ PaddleOCR 流水线初始化完成（全模块 TensorRT FP16）")
    print(f"  ⚡ 预期速度：60-100 页/分钟")
    return pipeline


def pdf_to_images_optimized(pdf_path: str) -> List[Tuple[int, any]]:
    """PDF 转图片（CPU 并行）"""
    import fitz
    import numpy as np
    import cv2
    
    images = []
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    
    print(f"  PDF 共 {total_pages} 页，CPU {PERF_CONFIG['cpu_workers']} 线程并行解码...")
    
    def decode_page(page_num):
        page = doc[page_num]
        mat = fitz.Matrix(PERF_CONFIG["dpi"] / 72, PERF_CONFIG["dpi"] / 72)
        pix = page.get_pixmap(matrix=mat)
        
        if pix.width > PERF_CONFIG["image_max_size"] or pix.height > PERF_CONFIG["image_max_size"]:
            scale = PERF_CONFIG["image_max_size"] / max(pix.width, pix.height)
            new_width = int(pix.width * scale)
            new_height = int(pix.height * scale)
            img = np.frombuffer(pix.tobytes("png"), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (new_width, new_height))
        else:
            img = np.frombuffer(pix.tobytes("png"), np.uint8)
            img = cv2.imdecode(img, cv2.IMREAD_COLOR)
        
        return (page_num + 1, img)
    
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=PERF_CONFIG["cpu_workers"]) as executor:
        results = list(executor.map(decode_page, range(total_pages)))
    
    images = sorted(results, key=lambda x: x[0])
    doc.close()
    
    print(f"  ✓ CPU 解码完成，{total_pages}页已加载到内存")
    return images


def process_page(img, page_num, pipeline) -> Dict:
    """处理单页（带表格检测）"""
    result = pipeline.predict(img)
    
    page_data = {
        "page": page_num,
        "text": "",
        "text_lines": [],
        "scores": [],
        "avg_score": 0.0,
        "is_table": False,
        "table_html": ""
    }
    
    for res in result:
        if "rec_texts" in res:
            texts = res.get("rec_texts", [])
            scores = res.get("rec_scores", [])
            boxes = res.get("rec_boxes", [])
            
            page_data["text"] = "\n".join(texts)
            page_data["scores"] = scores
            if scores:
                page_data["avg_score"] = sum(scores) / len(scores)
            
            # 保存带位置的文本行
            for i, text in enumerate(texts):
                box = boxes[i] if i < len(boxes) else None
                page_data["text_lines"].append({
                    "text": text,
                    "score": scores[i] if i < len(scores) else 0.0,
                    "box": box.tolist() if box is not None else None
                })
        
        # 检测表格（PP-StructureV3 输出）
        if "table_result" in res:
            table_res = res.get("table_result", {})
            if "html" in table_res:
                page_data["is_table"] = True
                page_data["table_html"] = table_res["html"]
    
    return page_data


def sort_text_lines_ancient(text_lines: List[Dict], img_width: int) -> List[Dict]:
    """
    古籍文本排序：从右到左、从上到下
    1. 按 Y 坐标分组（同一行）
    2. 每组内按 X 坐标从右到左排序
    
    box 格式：[[x1,y1], [x2,y2], [x3,y3], [x4,y4]] 或 [x1,y1,x2,y2,x3,y3,x4,y4]
    """
    if not text_lines:
        return text_lines
    
    def get_y_center(box):
        """获取文本框的 Y 中心坐标"""
        if not box:
            return 0
        if isinstance(box[0], list):
            # box 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            y_coords = [point[1] for point in box]
            return sum(y_coords) / len(y_coords)
        else:
            # box 是 [x1,y1,x2,y2,x3,y3,x4,y4]
            y_coords = [box[i] for i in range(1, len(box), 2)]
            return sum(y_coords) / len(y_coords)
    
    def get_x_right(box):
        """获取文本框的右侧 X 坐标（从右到左排序用）"""
        if not box:
            return 0
        if isinstance(box[0], list):
            # box 是 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in box]
            return max(x_coords)
        else:
            # box 是 [x1,y1,x2,y2,x3,y3,x4,y4]
            x_coords = [box[i] for i in range(0, len(box), 2)]
            return max(x_coords)

    # 按 Y 坐标排序
    sorted_by_y = sorted(text_lines, key=lambda x: get_y_center(x["box"]) if x["box"] else 0)
    
    # 分组（Y 坐标相近的为一行）
    lines_groups = []
    current_group = []
    current_y = -1
    y_threshold = 30
    
    for item in sorted_by_y:
        if item["box"]:
            y = get_y_center(item["box"])
            if current_y < 0 or abs(y - current_y) > y_threshold:
                if current_group:
                    lines_groups.append(current_group)
                current_group = [item]
                current_y = y
            else:
                current_group.append(item)
    
    if current_group:
        lines_groups.append(current_group)
    
    # 每组内按 X 坐标从右到左排序
    sorted_lines = []
    for group in lines_groups:
        sorted_group = sorted(group, key=lambda x: -get_x_right(x["box"]) if x["box"] else 0)
        sorted_lines.extend(sorted_group)
    
    return sorted_lines


def format_page_ancient(page_data: Dict) -> str:
    """格式化古籍页面输出（从右到左、框线版面，支持表格）"""
    output_lines = []

    border_width = 80
    output_lines.append("╔" + "═" * border_width + "╗")
    output_lines.append("║" + f" 第 {page_data['page']} 页".center(border_width) + "║")
    output_lines.append("╠" + "═" * border_width + "╣")

    # 检测并处理表格
    if page_data.get("is_table", False) and page_data.get("table_html"):
        output_lines.append("║ 【表格区域】")
        output_lines.append("║ " + page_data["table_html"])
        output_lines.append("║")
    elif page_data.get("text_lines"):
        sorted_lines = sort_text_lines_ancient(page_data["text_lines"], page_data["img_width"])

        for line in sorted_lines:
            text = line["text"]
            if text.strip():
                output_lines.append("║ " + text)

    # 页面底边框
    output_lines.append("╚" + "═" * border_width + "╝")

    return "\n".join(output_lines)


def format_page_modern(page_data: Dict) -> str:
    """格式化现代页面输出（标准 Markdown）"""
    output_lines = []
    
    output_lines.append(f"## 第 {page_data['page']} 页\n")
    
    if page_data["text"]:
        output_lines.append(page_data["text"])
    else:
        output_lines.append("*(无识别内容)*")
    
    output_lines.append("")
    output_lines.append("---")
    output_lines.append("")
    
    return "\n".join(output_lines)


def save_quality_report(pdf_name: str, pages_data: List[Dict], doc_type: str, output_dir: str):
    """保存质量报告到单独文件"""
    import json
    
    report_file = Path(output_dir) / f"{pdf_name}_quality_report.json"
    
    total_pages = len(pages_data)
    avg_confidence = sum(p["avg_score"] for p in pages_data) / total_pages if total_pages > 0 else 0
    min_confidence = min(p["avg_score"] for p in pages_data) if pages_data else 0
    high_confidence_count = sum(1 for p in pages_data if p["avg_score"] >= 0.90)
    medium_confidence_count = sum(1 for p in pages_data if 0.70 <= p["avg_score"] < 0.90)
    low_confidence_count = sum(1 for p in pages_data if p["avg_score"] < 0.70)
    
    report = {
        "pdf_name": pdf_name,
        "report_time": datetime.now().isoformat(),
        "version": VERSION,
        "doc_type": doc_type,
        "total_pages": total_pages,
        "quality_stats": {
            "avg_confidence": round(avg_confidence, 4),
            "min_confidence": round(min_confidence, 4),
            "high_confidence_pages": high_confidence_count,
            "medium_confidence_pages": medium_confidence_count,
            "low_confidence_pages": low_confidence_count,
            "high_confidence_ratio": round(high_confidence_count / total_pages * 100, 1) if total_pages > 0 else 0
        },
        "low_confidence_pages": [
            {"page": p["page"], "score": round(p["avg_score"], 4)}
            for p in pages_data if p["avg_score"] < 0.70
        ]
    }
    
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  ✓ 质量报告：{report_file.name}")
    
    print(f"\n  ╔════════════════════════════════════════════════════════╗")
    print(f"  ║  质量报告摘要                                            ║")
    print(f"  ╚════════════════════════════════════════════════════════╝")
    print(f"    文档类型：{'古籍' if doc_type == 'ancient' else '现代'}")
    print(f"    平均置信度：{avg_confidence:.4f}")
    print(f"    最低置信度：{min_confidence:.4f}")
    print(f"    ≥0.90 页面：{high_confidence_count}/{total_pages} ({high_confidence_count/total_pages*100:.1f}%)")
    print(f"    0.70-0.90 页面：{medium_confidence_count}/{total_pages} ({medium_confidence_count/total_pages*100:.1f}%)")
    print(f"    <0.70 页面：{low_confidence_count}/{total_pages} ({low_confidence_count/total_pages*100:.1f}%)")


def process_pdf_smart(pdf_path: str, output_dir: str, report_dir: str, pipeline) -> Tuple[str, float]:
    """处理单个 PDF（智能识别）"""
    start_time = time.time()
    pdf_name = Path(pdf_path).stem
    output_file = Path(output_dir) / f"{pdf_name}_ocr.md"
    
    print(f"\n处理：{Path(pdf_path).name}")
    
    print(f"  CPU 预取页面到内存...")
    prefetch_start = time.time()
    images = pdf_to_images_optimized(pdf_path)
    total_pages = len(images)
    prefetch_time = time.time() - prefetch_start
    print(f"  ✓ 预取完成，耗时{prefetch_time:.1f}秒")
    
    # 智能识别文档类型（使用第一页）
    print(f"  正在识别文档类型...")
    first_page_data = process_page(images[0][1], 1, pipeline)
    doc_type = detect_document_type(pdf_name, first_page_data["text"])
    
    if doc_type == "ancient":
        print(f"  ✓ 识别为【古籍文档】→ 使用从右到左、框线版面")
    else:
        print(f"  ✓ 识别为【现代文档】→ 使用标准 Markdown 格式")
    
    page_cache = PageCache(max_pages=PERF_CONFIG["prefetch_pages"])
    for i, (page_num, img) in enumerate(images):
        page_cache.put((pdf_name, page_num), img)
    
    pages_data = []

    print(f"\n  开始 OCR 识别...")

    for page_num, img in images:
        result = process_page(img, page_num, pipeline)
        pages_data.append(result)

        # 每 20 页检查并清理显存（智能监控）
        if page_num % 20 == 0:
            mem_mb = check_and_clear_gpu_memory(threshold_mb=14000)
            if mem_mb > 10000:
                print(f"  📊 显存占用：{mem_mb/1024:.1f}GB")

        if page_num % 10 == 0:
            elapsed = time.time() - start_time
            ppm = page_num / (elapsed / 60) if elapsed > 0 else 0
            print(f"  进度：{page_num}/{total_pages} | GPU 速度：{ppm:.1f}页/分钟 | 置信度：{result['avg_score']:.3f}")
            sys.stdout.flush()

    # 处理完成后清理显存
    check_and_clear_gpu_memory(threshold_mb=10000)
    
    page_cache.clear()
    images.clear()

    # 保存 OCR 文档（精简格式，无文件头）
    with open(output_file, "w", encoding="utf-8") as f:
        for page_data in pages_data:
            if doc_type == "ancient":
                f.write(format_page_ancient(page_data))
            else:
                f.write(format_page_modern(page_data))
            f.write("\n\n")

    # 保存质量报告
    save_quality_report(pdf_name, pages_data, doc_type, report_dir)
    
    elapsed = time.time() - start_time
    ppm = total_pages / (elapsed / 60) if elapsed > 0 else 0
    
    print(f"  ✓ 输出：{output_file.name}")
    print(f"  ✓ 速度：{ppm:.1f}页/分钟")
    
    return str(output_file), ppm


def load_state() -> Dict:
    """加载处理状态"""
    import json
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_state(state: Dict):
    """保存处理状态"""
    import json
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def scan_pdfs(folder: str) -> List[str]:
    """扫描 PDF"""
    pdf_files = []
    for f in os.listdir(folder):
        fp = os.path.join(folder, f)
        if os.path.isfile(fp) and f.lower().endswith(".pdf"):
            pdf_files.append(fp)
    return sorted(pdf_files)


def main():
    """主函数"""
    print("="*70)
    print(f"🚀 PDF 批量 OCR - {VERSION} 智能识别版")
    print(f"调整：{VERSION_NOTE}")
    print(f"目标：{TARGET_SPEED} 页/分钟")
    print("="*70)
    
    import paddle
    if paddle.is_compiled_with_cuda():
        gpu_name = paddle.device.cuda.get_device_name(0)
        gpu_mem = paddle.device.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} | 显存：{gpu_mem:.1f}GB")
    else:
        print("⚠️  运行模式：CPU")
    
    cpu_count = multiprocessing.cpu_count()
    print(f"✅ CPU: {cpu_count} 核心 | {PERF_CONFIG['cpu_workers']} 线程解码")
    print(f"✅ 内存：96GB | 预取缓存：{PERF_CONFIG['prefetch_pages']} 页")
    print("="*70)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    
    state = load_state()
    processed = state.get("processed", [])
    
    print(f"\n📂 扫描目录：{INPUT_DIR}")
    pdf_files = scan_pdfs(INPUT_DIR)
    
    new_files = [f for f in pdf_files if f not in processed]
    
    if not new_files:
        print("⚠️  未找到新 PDF 文件")
        return
    
    print(f"📊 共发现 {len(pdf_files)} 个 PDF 文件")
    print(f"📊 待处理 {len(new_files)} 个 PDF 文件")
    
    print("\n初始化 PaddleOCR 流水线...")
    pipeline = init_ocr_pipeline()
    
    print(f"\n开始批量处理...")
    for idx, pdf in enumerate(new_files, 1):
        print(f"\n[{idx}/{len(new_files)}]")
        try:
            output, speed = process_pdf_smart(pdf, OUTPUT_DIR, REPORT_DIR, pipeline)
            
            processed.append(pdf)
            state["processed"] = processed
            state["last_updated"] = datetime.now().isoformat()
            state["last_file"] = pdf
            state["last_speed"] = speed
            state["version"] = VERSION
            save_state(state)
            
        except Exception as e:
            print(f"  ❌ 错误：{e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("🎉 批量处理完成！")
    print(f"📁 OCR 输出目录：{OUTPUT_DIR}")
    print(f"📁 质量报告目录：{REPORT_DIR}")
    print("="*70)


if __name__ == "__main__":
    main()
