import subprocess
import os
import glob
import json
import re
import argparse
from datetime import datetime
from openai import OpenAI
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from token_tracker import tracker 
from logger import setup_logger

# ==========================================
# 1. 大模型配置区域 (从 .env 自动加载，彻底消除硬编码)
# ==========================================
class AppSettings(BaseSettings):
    api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    base_url: str = Field(..., alias="DEEPSEEK_BASE_URL")
    model_name: str = Field(..., alias="DEEPSEEK_MODEL_NAME")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True
    )

# 实例化配置，如果 .env 缺失配置项，程序会在此处直接报错 (Fail-Fast)
settings = AppSettings()


# ==========================================
# 2. 核心功能函数
# ==========================================
def run_mineru(pdf_path: str, output_dir: str) -> bool:
    """
    调用 MinerU 解析单个 PDF 文件
    :return: 解析成功返回 True，失败返回 False
    """
    logger=setup_logger("Reader.MinerU")
    logger.info(f"\n  [MinerU] 正在解析: {os.path.basename(pdf_path)}")
    logger.info(f"  [MinerU] 输出目录: {os.path.basename(output_dir)}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建 MinerU 命令行指令
    cmd = [
        "mineru", 
        "-p", pdf_path, 
        "-o", output_dir, 
        "-b", "pipeline"
    ]
    
    try:
        # 【核心修改】：
        # 1. 去掉了 capture_output=True 和 text=True、encoding='utf-8'
        # 2. 让 MinerU 的进度条和日志直接打印在当前终端，体验最好，且彻底解决编码报错
        subprocess.run(cmd, check=True) 
        
        logger.info(f"  ✅ [MinerU] 解析成功: {os.path.basename(pdf_path)}")
        return True
        
    except subprocess.CalledProcessError as e:
        # 如果解析失败，直接打印退出码，不再去读取 stderr 避免 NoneType 报错
        logger.error(f"  ❌ [MinerU] 解析失败，退出码: {e.returncode}")
        return False
    except FileNotFoundError:
        logger.error("  ❌ [错误] 找不到 'mineru' 命令，请确保你已激活 mineru_new 虚拟环境。")
        return False
'''
def run_mineru(pdf_path: str, output_dir: str) -> bool:
    """
    调用 MinerU 解析单个 PDF 文件
    :return: 解析成功返回 True，失败返回 False
    """
    print(f"  [MinerU] 正在解析: {os.path.basename(pdf_path)}")
    
    try:
        cmd = [
            "mineru", 
            "-p", pdf_path, 
            "-o", output_dir, 
            "-b", "pipeline"
        ]
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [MinerU] 解析失败 {os.path.basename(pdf_path)}: {e.stderr[:100]}")
        return False
    except FileNotFoundError:
        print("  [错误] 找不到 'mineru' 命令，请确保已激活 mineru_new 虚拟环境")
        return False

'''
def find_md_file(output_dir: str, pdf_filename: str) -> str:
    """
    在输出目录中查找与 PDF 对应的 Markdown 文件
    :return: Markdown 文件内容，失败返回 None
    """
    # MinerU 会为每个 PDF 创建一个子文件夹
    pdf_name_without_ext = Path(pdf_filename).stem
    
    # 在输出目录中查找对应的 md 文件
    md_pattern = os.path.join(output_dir, "**", f"{pdf_name_without_ext}*.md")
    md_files = glob.glob(md_pattern, recursive=True)
    
    if md_files:
        with open(md_files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    # 如果没找到精确匹配的，就找任意 md 文件
    md_files = glob.glob(os.path.join(output_dir, "**/*.md"), recursive=True)
    if md_files:
        with open(md_files[0], 'r', encoding='utf-8') as f:
            return f.read()
    
    return None


def clean_llm_json_response(response_text: str) -> dict:
    """清洗大模型返回的 JSON 字符串"""
    cleaned_text = re.sub(r'^```json\s*', '', response_text.strip())
    cleaned_text = re.sub(r'\s*```$', '', cleaned_text.strip())
    
    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError as e:
        logger = setup_logger("Reader.LLM")
        logger.error(f"  [LLM] JSON 解析失败: {e}")
        return None


def extract_structured_info(markdown_content: str, pdf_filename: str) -> dict:
    """
    调用 LLM 从 Markdown 中提取结构化信息
    :return: 包含论文信息的字典
    """
    logger = setup_logger("Reader.LLM")
    logger.info(f"\n  [LLM] 正在从 Markdown 中提取结构化信息...")
    # 【修改点】：使用从 .env 加载的 settings 初始化客户端
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    
    prompt = f"""
你是一位专业的学术文献分析专家。请阅读以下 PDF 解析出的 Markdown 文本，提取关键信息。

【严格规则】：
1. 只能基于提供的文本提取信息，严禁捏造（零幻觉）
2. 如果找不到某项信息，填写 "未提及"
3. 必须严格按照 JSON 格式输出，不要有任何额外文字

【提取结构】：
{{
    "filename": "{os.path.basename(pdf_filename)}",
    "title": "论文标题",
    "authors": ["作者1", "作者2"],
    "year": "发表年份",
    "research_background": "研究背景与动机 (100字以内)",
    "core_method": "核心方法/模型及关键技术 (200字以内)",
    "key_experiments": "主要实验设置与核心指标 (200字以内)",
    "limitations": "作者提到的局限性",
    "my_notes": "这篇论文最值得借鉴的亮点 (100字以内)"
}}

【Markdown 内容】：
{markdown_content[:12000]}
"""

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "你是学术分析助手，只输出合法 JSON"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        raw_content = response.choices[0].message.content
        # 记录 token 使用情况
        # 记录 token 使用情况
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        tracker.record(
            agent="Reader",
            model=settings.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            operation=f"提取论文: {os.path.basename(pdf_filename)}"
        )        
        '''        
        if hasattr(response, 'usage') and response.usage:
            tracker.record(
                agent="Reader",
                model=settings.model_name,
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                operation=f"提取论文: {os.path.basename(pdf_filename)}"
            )        
        '''
        structured_data = clean_llm_json_response(raw_content)
        
        if structured_data:
            logger.info(f"  ✓ 成功提取: {os.path.basename(pdf_filename)}")
        
        return structured_data
        
    except Exception as e:
        logger.error(f"  [LLM] 调用失败 {os.path.basename(pdf_filename)}: {e}")
        return None


def process_single_pdf(pdf_path: str, mineru_output_dir: str) -> dict:
    """
    处理单个 PDF 文件：解析 -> 提取信息
    :return: 结构化信息字典
    """
    logger = setup_logger("Reader")
    logger.info(f"\n  [Reader] 正在处理 PDF: {os.path.basename(pdf_path)}")
    # 1. 调用 MinerU 解析
    if not run_mineru(pdf_path, mineru_output_dir):
        return None
    
    # 2. 查找并读取 Markdown
    md_content = find_md_file(mineru_output_dir, pdf_path)
    if not md_content:
        logger.error(f"  [错误] 未找到 Markdown 文件: {os.path.basename(pdf_path)}")
        return None
    
    # 3. 调用 LLM 提取结构化信息
    return extract_structured_info(md_content, pdf_path)


def sanitize_filename(filename: str) -> str:
    """清理文件名，移除非法字符"""
    # 移除或替换非法字符
    invalid_chars = '<>:"/\\|？*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename.strip()


def main():
    print("\n" + "="*60)
    print("📚 学术文献综述自动生成系统 - 批量文献解析模块")
    print("="*60)
    
    # ==========================================
    # 用户输入配置
    # ==========================================
    
    # 1. 输入综述主题
    print("\n[步骤1] 请输入你的文献综述主题：")
    print("  例如：大模型在医疗诊断中的应用")
    topic = input("  主题 > ").strip()
    
    if not topic:
        topic = "literature_review"
        print("  未输入主题，使用默认名称")
    
    # 清理主题字符串，用于生成文件名
    safe_topic = sanitize_filename(topic)[:50]  # 限制长度
    
    # 2. 输入 PDF 文件路径或目录
    print(f"\n[步骤2] 请输入 PDF 文件所在的路径：")
    print("  可以是单个 PDF 文件，也可以是包含多个 PDF 的文件夹")
    print("  例如：C:\\Users\\LOVE\\Desktop\\papers")
    input_path = input("  路径 > ").strip()
    
    # 处理路径中的引号（用户可能复制粘贴带引号的路径）
    input_path = input_path.strip('"\'')
    
    if not os.path.exists(input_path):
        print(f"\n[错误] 路径不存在: {input_path}")
        return
    
    # 3. 收集所有待处理的 PDF 文件
    pdf_files = []
    if os.path.isfile(input_path):
        # 单个文件
        if input_path.lower().endswith('.pdf'):
            pdf_files.append(input_path)
        else:
            print(f"\n[错误] 不是 PDF 文件: {input_path}")
            return
    else:
        # 目录：递归查找所有 PDF
        pdf_files = glob.glob(os.path.join(input_path, "**/*.pdf"), recursive=True)
        
        if not pdf_files:
            print(f"\n[错误] 在目录中未找到任何 PDF 文件: {input_path}")
            return
    
    print(f"\n[信息] 找到 {len(pdf_files)} 个 PDF 文件待处理")
    
    # 4. 配置输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = f"./output_{safe_topic}_{timestamp}"
    mineru_output_dir = os.path.join(base_output_dir, "mineru_parsed")
    
    os.makedirs(mineru_output_dir, exist_ok=True)
    print(f"[信息] 输出目录: {base_output_dir}")
    
    # ==========================================
    # 批量处理 PDF
    # ==========================================
    
    print("\n" + "="*60)
    print("开始批量解析文献...")
    print("="*60)
    
    all_papers_info = []
    success_count = 0
    fail_count = 0
    
    for idx, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{idx}/{len(pdf_files)}] 处理进度:")
        
        paper_info = process_single_pdf(pdf_path, mineru_output_dir)
        
        if paper_info:
            all_papers_info.append(paper_info)
            success_count += 1
        else:
            fail_count += 1
            print(f"  ✗ 处理失败: {os.path.basename(pdf_path)}")
    
    # ==========================================
    # 保存结果
    # ==========================================
    
    print("\n" + "="*60)
    print("处理完成！保存结果...")
    print("="*60)
    
    # 1. 保存总的 JSON 文件（包含所有论文信息）
    total_json_path = os.path.join(base_output_dir, f"{safe_topic}_all_papers.json")
    
    review_metadata = {
        "topic": topic,
        "created_at": datetime.now().isoformat(),
        "total_papers": len(pdf_files),
        "success_count": success_count,
        "fail_count": fail_count,
        "papers": all_papers_info
    }
    
    with open(total_json_path, 'w', encoding='utf-8') as f:
        json.dump(review_metadata, f, indent=4, ensure_ascii=False)
    
    print(f"\n✓ 总文件已保存: {total_json_path}")
    
    # 2. 为每篇论文保存单独的 JSON 文件
    individual_dir = os.path.join(base_output_dir, "individual_papers")
    os.makedirs(individual_dir, exist_ok=True)
    
    for paper in all_papers_info:
        paper_filename = paper.get('filename', 'unknown').replace('.pdf', '.json')
        individual_path = os.path.join(individual_dir, paper_filename)
        
        with open(individual_path, 'w', encoding='utf-8') as f:
            json.dump(paper, f, indent=4, ensure_ascii=False)
    
    print(f"✓ 单篇论文 JSON 已保存至: {individual_dir}")
    
    # 3. 生成简单的统计报告
    report_path = os.path.join(base_output_dir, "processing_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"文献综述主题: {topic}\n")
        f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总论文数: {len(pdf_files)}\n")
        f.write(f"成功解析: {success_count}\n")
        f.write(f"解析失败: {fail_count}\n\n")
        f.write("论文列表:\n")
        for i, paper in enumerate(all_papers_info, 1):
            f.write(f"{i}. {paper.get('title', '未知标题')}\n")
            f.write(f"   作者: {', '.join(paper.get('authors', []))}\n")
            f.write(f"   年份: {paper.get('year', '未知')}\n\n")
    
    print(f"✓ 处理报告已保存: {report_path}")
    
    # ==========================================
    # 打印摘要
    # ==========================================
    
    print("\n" + "="*60)
    print("📊 处理结果摘要")
    print("="*60)
    print(f"主题: {topic}")
    print(f"成功解析: {success_count}/{len(pdf_files)} 篇")
    print(f"输出目录: {os.path.abspath(base_output_dir)}")
    print(f"\n下一步：你可以使用 {total_json_path} 文件进行综述撰写")
    print("="*60 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[提示] 用户中断程序，已处理的数据已保存")
    except Exception as e:
        print(f"\n[错误] 程序异常: {e}")
        import traceback
        traceback.print_exc()