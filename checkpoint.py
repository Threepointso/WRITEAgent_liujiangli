"""
Reader 逐篇解析进度管理
每解析完一篇 PDF，记录文件名；下次读取时跳过已完成的。
"""

import json
import os

READER_PROGRESS = ".reader_progress.json"


def save_reader_progress(output_dir: str, completed: list[str]):
    """保存 Reader 已解析完成的 PDF 文件名列表"""
    path = os.path.join(output_dir, READER_PROGRESS)
    os.makedirs(output_dir, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({"completed": completed}, f, indent=2)
    # 不打印，避免刷屏


def load_reader_progress(output_dir: str) -> set[str]:
    """加载 Reader 已解析完成的 PDF 文件名集合"""
    path = os.path.join(output_dir, READER_PROGRESS)
    if not os.path.exists(path):
        return set()
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return set(json.load(f).get("completed", []))
    except (json.JSONDecodeError, IOError):
        return set()


def clear_reader_progress(output_dir: str):
    """Reader 全部完成后清理进度文件"""
    path = os.path.join(output_dir, READER_PROGRESS)
    if os.path.exists(path):
        os.remove(path)