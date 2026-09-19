"""
日志模块
统一管理所有 Agent 的日志输出，同时写入文件和终端
"""

import logging
import os
import sys
from datetime import datetime

# 日志目录
LOG_DIR = "./logs"
os.makedirs(LOG_DIR, exist_ok=True)

# 日志文件名（按日期）
log_filename = os.path.join(LOG_DIR, f"pipeline_{datetime.now().strftime('%Y%m%d')}.log")


def setup_logger(name: str = "pipeline") -> logging.Logger:
    """
    创建或获取一个日志器
    用法: logger = setup_logger("Reader")
          logger.info("开始解析 PDF")
          logger.error("解析失败: %s", pdf_name)
    """
    logger = logging.getLogger(name)
    
    # 防止重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)
    
    # --- 格式定义 ---
    file_fmt = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_fmt = logging.Formatter("%(message)s")
    
    # --- 文件 Handler（记录所有级别） ---
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)
    
    # --- 终端 Handler（只显示 INFO 及以上，保持界面干净） ---
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)
    
    return logger