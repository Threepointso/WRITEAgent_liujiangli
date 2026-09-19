import os
import json
import glob
import re
import asyncio
from typing import Optional, Tuple, Dict, List

from openai import AsyncOpenAI
from pydantic import BaseModel, Field, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict
from token_tracker import tracker  
from logger import setup_logger

# ==========================================
# 1. 配置中心（保持不变）
# ==========================================
class AppSettings(BaseSettings):
    """从 .env 自动加载配置"""
    api_key: str = Field(..., alias="DEEPSEEK_API_KEY")
    base_url: str = Field(..., alias="DEEPSEEK_BASE_URL")
    model_name: str = Field(..., alias="DEEPSEEK_MODEL_NAME")
    output_tex_path: str = Field(default="./review_output.tex", alias="OUTPUT_TEX_PATH")

    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore',
        populate_by_name=True
    )

# ==========================================
# 2. 数据模型（保持不变）
# ==========================================
class Paper(BaseModel):
    """论文数据模型"""
    title: str = "未知标题"
    authors: List[str] = []
    year: Optional[str | int] = "未知"
    research_background: str = "未提及"
    core_method: str = "未提及"
    key_experiments: str = "未提及"
    limitations: str = "未提及"

# ==========================================
# 3. 核心功能函数
# ==========================================

async def find_latest_json(topic: str) -> Optional[str]:
    """根据主题自动查找 all_papers.json（保持不变）"""
    logger = setup_logger()
    logger.info(f"\n🔍 [Writer] 正在根据主题 '{topic}' 查找文献库...")
    
    def _search_files():
        pattern = f"./output_*{topic}*/*{topic}*_all_papers.json"
        files = glob.glob(pattern)
        if not files:
            logger.warning("   ⚠️ 未找到精确匹配的文件，尝试查找最近的解析结果...")
            pattern = "./output_*/**/*_all_papers.json"
            files = glob.glob(pattern, recursive=True)
        return files

    files = await asyncio.to_thread(_search_files)
    
    if not files:
        logger.error("❌ 错误：找不到任何 all_papers.json 文件！请先运行 Reader Agent。")
        return None
        
    latest_file = max(files, key=os.path.getmtime)
    logger.info(f"   ✅ 找到文献库: {os.path.abspath(latest_file)}")
    return latest_file


async def load_papers_and_build_whitelist(json_path: str) -> Tuple[str, Dict[str, str], str]:
    """加载论文数据并构建白名单（保持不变）"""
    logger = setup_logger("Writer")
    logger.info(f"\n📂 [Writer] 正在加载文献库...")
    
    def _read_and_parse():
        with open(json_path, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    data = await asyncio.to_thread(_read_and_parse)
    
    papers = data.get("papers", [])
    if not papers:
        logger.error("❌ 文献库为空！")
        return None, None, None

    topic = data.get("topic", "未知主题")
    logger.info(f"   📌 综述主题: {topic}")

    whitelist = {}
    context_for_llm = []
    
    for i, paper_dict in enumerate(papers, 1):
        paper = Paper(**paper_dict) 
        cite_id = f"paper_{i}"
        whitelist[cite_id] = paper.title
        
        ctx = f"""
=== 文献 [{cite_id}] ===
标题: {paper.title}
作者: {', '.join(paper.authors)}
年份: {paper.year}
研究背景: {paper.research_background}
核心方法: {paper.core_method}
实验与指标: {paper.key_experiments}
局限性: {paper.limitations}
"""
        context_for_llm.append(ctx)
        
    logger.info(f"   ✅ 成功加载 {len(papers)} 篇文献，白名单已构建。")
    return topic, whitelist, "\n".join(context_for_llm)


# ==========================================
# 表格生成相关函数（保持不变）
# ==========================================

def _json_to_latex_table(table_data: dict, table_type: str = "method") -> str:
    """
    将JSON表格数据转换为LaTeX表格
    """
    try:
        caption = table_data.get("caption", "对比表")
        columns = table_data.get("columns", [])
        rows = table_data.get("rows", [])
        
        if not columns or not rows:
            return ""
        
        col_format = "|l|" + "c|" * (len(columns) - 1)
        label_name = re.sub(r'[^a-zA-Z0-9]', '_', caption.lower())[:30]
        header = " & ".join([f"\\textbf{{{col}}}" for col in columns])
        
        latex_table = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{{caption}}}
\\label{{tab:{label_name}}}
\\begin{{tabular}}{{{col_format}}}
\\hline
{header} \\\\
\\hline
"""
        for row in rows:
            while len(row) < len(columns):
                row.append("-")
            cells = []
            for cell in row:
                if cell is None or cell == "" or cell == "null":
                    cells.append("-")
                elif isinstance(cell, str) and "%" in cell:
                    cells.append(cell.replace("%", "\\%"))
                else:
                    cells.append(str(cell))
            latex_table += " & ".join(cells) + " \\\\\n\\hline\n"
        
        latex_table += """\\end{tabular}
\\end{table}
"""
        return latex_table
        
    except Exception as e:
        logger = setup_logger("Writer")
        logger.error(f"   ⚠️ 表格转换失败: {e}")
        return ""


# ==========================================
# 主生成函数：分块策略 C
# ==========================================

def generate_latex_review(topic: str, papers_info: list, writing_req: str = "") -> Tuple[Optional[str], dict]:
    """
    调用LLM生成LaTeX格式的综述
    【策略C】：分类 → 逐节生成 → 每节只传对应论文 + 携带前文摘要
    """
    logger = setup_logger("Writer")
    logger.info(f"\n✍️ [Writer] 正在撰写关于《{topic}》的LaTeX综述（策略C：分块生成）...")
    
    # 1. 构建白名单和Context
    whitelist = {}
    context_for_llm = []
    for i, paper_dict in enumerate(papers_info, 1):
        paper = Paper(**paper_dict)
        cite_id = f"paper_{i}"
        whitelist[cite_id] = paper.title
        ctx = f"""
=== 文献 [{cite_id}] ===
标题: {paper.title}
作者: {', '.join(paper.authors)}
年份: {paper.year}
研究背景: {paper.research_background}
核心方法: {paper.core_method}
实验与指标: {paper.key_experiments}
局限性: {paper.limitations}
"""
        context_for_llm.append(ctx)
    
    settings = AppSettings()
    req_text = f"\n【用户额外写作要求】：\n{writing_req}\n" if writing_req else ""

    # ==========================================
    # 分块生成函数
    # ==========================================

    async def _generate_tables(client) -> dict:
        """
        Step 1: 轻量调用，只传标题+方法，将论文分为 2-4 类
        确保每个后续章节的 LLM 调用只处理 5-15 篇论文
        """
        logger.info("   📊 正在对论文进行技术分类（轻量调用）...")
        summaries = []
        for i, pd_ in enumerate(papers_info):
            p = Paper(**pd_)
            summaries.append(f"[{i+1}] 标题: {p.title} | 方法: {p.core_method[:80]}")
        cls_prompt = f"""你是一位学术分析专家。分析以下论文，按核心技术路线分为 2-4 个类别。

论文列表：
{chr(10).join(summaries)}

输出 JSON：
{{"categories": {{"分类名1": [论文编号], "分类名2": [论文编号]}}}}

注意：论文编号从 0 开始。"""
        resp = await client.chat.completions.create(
            model=settings.model_name,
            messages=[{"role": "user", "content": cls_prompt}],
            temperature=0.2
        )
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0
        tracker.record("Writer", settings.model_name, prompt_tokens, completion_tokens, "论文分类")
        raw = resp.choices[0].message.content.strip()
        raw = re.sub(r'^```json\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        try:
            result = json.loads(raw)
            cat = result.get("categories", {})
            valid_indices = set(range(len(papers_info)))
            for k in list(cat.keys()):
                cat[k] = [i for i in cat[k] if i in valid_indices]
            logger.info(f"   ✅ 论文分类完成: {list(cat.keys())}")
            return cat
        except Exception:
            logger.warning("   ⚠️ 论文分类失败，使用默认均分")
            n = len(papers_info)
            mid = n // 2
            return {"第一组": list(range(mid)), "第二组": list(range(mid, n))}

    async def _generate_text_content(client, title: str, instruction: str,
                                       indices: list[int],
                                       prev_sections: list[str]) -> Optional[str]:
        """
        Step 2: 生成单个章节，只携带 indices 对应的论文 + 前文片段（≤1500字符）
        """
        logger.info(f"   📝 正在生成章节: {title}（{len(indices)} 篇论文）...")
        subset = "\n".join([context_for_llm[i] for i in indices])
        prev_text = ""
        if prev_sections:
            combined = "\n\n".join(prev_sections)
            prev_text = f"\n【前面已完成的章节（请保持风格一致，避免重复）】：\n{combined[-1500:]}\n"
        sec_prompt = f"""根据以下【文献子集】，撰写 LaTeX 章节。{prev_text}

【章节标题】: \\section{{{title}}}
【写作要求】: {instruction}
{req_text}
【零幻觉规则】: 只能引用提供的文献 ID（\\cite{{paper_X}}），严禁捏造！

【文献子集】:
{subset}

直接输出 LaTeX 代码，不要包含 ``` 标记："""
        resp = await client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "你是一个严谨的学术写作助手，严格遵守零幻觉引用规则。"},
                {"role": "user", "content": sec_prompt}
            ],
            temperature=0.3
        )
        prompt_tokens = resp.usage.prompt_tokens if resp.usage else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage else 0
        tracker.record("Writer", settings.model_name, prompt_tokens, completion_tokens, f"生成章节: {title}")
        content = resp.choices[0].message.content
        content = content.replace("```latex", "").replace("```", "").strip()
        # 引用校验
        valid_ids = set(whitelist.keys())
        def _replace(m):
            cid = m.group(1)
            if cid in valid_ids:
                return m.group(0)
            logger.warning(f"  ⚠️ 幻觉引用并拦截: \\cite{{{cid}}}")
            return f"[引用无效: {cid}]"
        return re.sub(r"\\cite\{([^}]+)\}", _replace, content)

    async def _generate_all() -> Optional[str]:
        """
        Step 3: 编排——分类→逐节生成→拼接，每节只携带相关论文 + 上文摘要
        """
        client = AsyncOpenAI(api_key=settings.api_key, base_url=settings.base_url)
        n = len(papers_info)
        all_idx = list(range(n))
        prev_sections = []

        # ----- 分类（轻量调用）-----
        categories = await _generate_tables(client)

        # ----- 引言（所有论文概述）-----
        intro = await _generate_text_content(client, "引言",
            "介绍该领域背景，引出综述目的和本文结构", all_idx, prev_sections)
        if not intro:
            return None
        prev_sections.append("[引言] " + intro[:200])

        # ----- 方法对比（每类一个子节，只传本类论文）-----
        method_parts = []
        for cat_name, indices in categories.items():
            sec = await _generate_text_content(client, cat_name,
                f"详细对比该类论文的核心方法和技术路线，突出异同", indices, prev_sections)
            if sec:
                method_parts.append(sec)
                prev_sections.append(f"[{cat_name}] " + sec[:200])

        # ----- 实验分析（每类取前 2 篇代表作）-----
        exp_idx = []
        for idx_list in categories.values():
            exp_idx.extend(idx_list[:2])
        if not exp_idx:
            exp_idx = all_idx[:4]
        experiments = await _generate_text_content(client, "实验与局限性分析",
            "总结实验表现、核心指标、共同挑战与局限性", exp_idx, prev_sections)
        if experiments:
            prev_sections.append("[实验与局限性] " + experiments[:200])

        # ----- 总结（所有论文高层概括）-----
        conclusion = await _generate_text_content(client, "总结与未来展望",
            "总结全文关键发现，指出开放问题与未来方向", all_idx, prev_sections)

        # ----- 拼接完整 LaTeX -----
        bib_items = [f"\\bibitem{{{cite_id}}} {title}." for cite_id, title in whitelist.items()]
        bib_section = "\\begin{thebibliography}{99}\n" + "\n".join(bib_items) + "\n\\end{thebibliography}"

        parts = [
            "\\section{引言}\n", intro or "",
            "\n\n\\section{核心方法对比}\n",
            "\n\n".join(method_parts) if method_parts else "",
            "\n\n\\section{实验与局限性分析}\n", experiments or "",
            "\n\n\\section{总结与未来展望}\n", conclusion or "",
            "\n\n", bib_section
        ]
        full = "\n".join(parts)
        full = re.sub(r'\n{3,}', '\n\n', full)
        logger.info("   ✅ LaTeX 综述分块生成并校验完毕！")
        return full

    # 同步执行异步任务
    latex_content = asyncio.run(_generate_all())
    return latex_content, whitelist


# ==========================================
# 4. 主程序入口（保持不变）
# ==========================================

async def _main_async():
    """内部异步主逻辑"""
    print("\n" + "="*60)
    print("✍️ Writer Agent - 学术综述LaTeX生成模块（分块策略C）")
    print("="*60)
    
    try:
        settings = AppSettings()
        print("✅ 配置加载成功，API Key已安全从.env注入。")
    except ValidationError as e:
        print(f"❌ 配置加载失败！请检查.env文件。\n{e}")
        return

    topic = input("\n[步骤1] 请输入你要撰写的综述主题 (例如: 大模型在存储物流中的应用): ").strip()
    if not topic:
        print("❌ 主题不能为空！")
        return
        
    json_path = await find_latest_json(topic)
    if not json_path:
        return
        
    real_topic, whitelist, context = await load_papers_and_build_whitelist(json_path)
    if not whitelist:
        return
        
    papers_data = [{"title": w, "authors": [], "year": "未知", "research_background": "无", "core_method": "无", "key_experiments": "无", "limitations": "无"} for w in whitelist.values()]
    latex_content, _ = generate_latex_review(real_topic, papers_data, "")
    
    if latex_content:
        def _write():
            with open(settings.output_tex_path, 'w', encoding='utf-8') as f:
                f.write(latex_content)
        await asyncio.to_thread(_write)
        
        print(f"\n🎉 综述已保存至: {os.path.abspath(settings.output_tex_path)}")
        print(" 下一步：运行Critic Agent检查是否存在幻觉引用！")

def main():
    """对外暴露的同步main函数"""
    asyncio.run(_main_async())

if __name__ == "__main__":
    main()