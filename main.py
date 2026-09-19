"""
Multi-Agent 综述生成工具箱
每个模块独立运行，无模式选择，无中间暂停提问
"""

import os
import sys
import time
import json
import re
import glob
from checkpoint import (
    save_reader_progress, load_reader_progress, clear_reader_progress,
)
from token_tracker import tracker
from logger import setup_logger
# ==========================================
# 导入模块
# ==========================================
try:
    from planner_agent import generate_search_strategy
    from searcher_agent import (
        search_arxiv, enrich_with_citations, filter_and_rank_papers,
        batch_download, save_search_metadata, hybrid_search
    )
    from reader_agent import process_single_pdf
    from writer_agent import generate_latex_review
    from critic_agent import extract_citations, verify_citations
    from coordinator import Coordinator
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)


def sanitize_filename(name: str) -> str:
    return re.sub(r'[<>:"/\\|?*]', '', name).replace(' ', '_')[:30]


def find_latest_dir(pattern: str) -> str | None:
    dirs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return dirs[0] if dirs else None


def find_latest_file(pattern: str) -> str | None:
    files = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return files[0] if files else None


# ==========================================
# 模块 1: Planner + Searcher（从头开始）
# ==========================================
def module_planner_searcher():
    """输入主题 → 规划检索式 → 下载论文到文件夹"""
    print("\n" + "=" * 55)
    print("  [1] 搜索并下载论文 (Planner + Searcher)")
    print("=" * 55)

    topic = input("\n  请输入研究主题 (中文即可):\n  > ").strip()
    if not topic:
        print("  ❌ 主题不能为空！"); return

    # --- Planner ---
    print("\n  -> 正在分析主题并生成检索式...")
    strategy = generate_search_strategy(topic)
    # Planner 现在返回 4 条检索式，默认用精确匹配
    print(f"  📋 多角度检索策略（精确/同义/相关/综述）")
    print(f"     🎯 精确: {strategy.get('exact_query', '')[:60]}")
    confirm = input("\n  是否使用此策略？(y=使用 / n=手动输入): ").strip().lower()
    final_query = strategy.get('exact_query', topic) if confirm != 'n' \
    else input("  请输入检索式: ").strip()
    while True:
        try:
            n = int(input("  下载数量: ").strip())
            if n > 0: break
        except: pass

    # --- Searcher（混合检索：arXiv + 本地向量库）---
    final_papers = hybrid_search(final_query, topic, n)
    if not final_papers:
        print("  ❌ 未找到相关论文。"); return
    dl = input(f"\n  筛选出 {len(final_papers)} 篇论文，是否下载？(y/n): ").strip().lower()
    if dl != 'y':
        print("  已取消下载。"); return

    safe = sanitize_filename(topic)
    ts = time.strftime("%Y%m%d_%H%M%S")
    download_dir = f"./papers_{safe}_{ts}"

    print(f"\n  -> 正在下载到: {download_dir}")
    paths = batch_download(final_papers, download_dir)
    save_search_metadata(final_papers, download_dir, topic)

    if paths:
        print(f"\n  ✅ 下载完成！共 {len(paths)} 篇")
        print(f"  📁 保存位置: {os.path.abspath(download_dir)}")
    else:
        print("  ❌ 所有论文下载均失败。")


# ==========================================
# 模块 2: Reader（指定 PDF 文件夹）
# ==========================================
def module_reader():
    """选择 PDF 文件夹 → 逐篇解析 → 输出 JSON"""
    print("\n" + "=" * 55)
    print("  [2] 解析 PDF → JSON (Reader)")
    print("=" * 55)

    latest = find_latest_dir("./papers_*")
    hint = f"\n  💡 最新文件夹: {latest}" if latest else ""
    print(hint)

    pdf_dir = input("\n  请输入 PDF 文件夹路径:\n  > ").strip()
    if not pdf_dir:
        print("  ❌ 路径不能为空！"); return
    if not os.path.isdir(pdf_dir):
        print(f"  ❌ 文件夹不存在: {pdf_dir}"); return

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdfs:
        print("  ❌ 该文件夹中没有 PDF 文件。"); return

    print(f"\n  📄 共发现 {len(pdfs)} 篇 PDF")

    # 自动创建输出目录（与下载目录对应）
    dirname = os.path.basename(pdf_dir).replace("papers_", "output_")
    output_dir = f"./{dirname}"
    json_path = os.path.join(output_dir, "all_papers_info.json")

    # 检测已解析进度
    all_papers = []
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            all_papers = json.load(f).get("papers", [])
        print(f"  📂 已有 {len(all_papers)} 篇解析结果")

    completed = load_reader_progress(output_dir)
    if completed:
        print(f"  📂 已记录 {len(completed)} 篇完成，将跳过已解析的 PDF")

    pending = [p for p in pdfs if os.path.basename(p) not in completed]
    if not pending:
        print("\n  ✅ 所有 PDF 均已解析完成！")
        print(f"  📁 {os.path.abspath(json_path)}")
        return

    confirm = input(f"\n  待解析 {len(pending)} 篇，是否开始？(y/n): ").strip().lower()
    if confirm != 'y':
        print("  已取消。"); return

    mineru_dir = os.path.join(output_dir, "mineru_parsed")
    os.makedirs(mineru_dir, exist_ok=True)

    for idx, pdf_path in enumerate(pending, 1):
        pdf_name = os.path.basename(pdf_path)
        print(f"\n  --- [{idx}/{len(pending)}] {pdf_name} ---")
        info = process_single_pdf(pdf_path, mineru_dir)
        if info:
            all_papers.append(info)
            completed.add(pdf_name)
            save_reader_progress(output_dir, list(completed))
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump({"papers": all_papers}, f, indent=4, ensure_ascii=False)
            print(f"  ✅ 进度: {len(completed)}/{len(pdfs)}")
    if all_papers:
        clear_reader_progress(output_dir)
        print(f"\n  ✅ 解析完成！共 {len(all_papers)} 篇")
        print(f"  📁 {os.path.abspath(json_path)}")
        
        # 自动存入本地向量库（供后续综述复用）
        try:
            from vector_store import vector_store
            # 读取 search_metadata.json 获取摘要信息
            # 找到对应的 papers_xxx 目录
            base_dir = os.path.dirname(output_dir)
            papers_dir = output_dir.replace('output_', 'papers_', 1)
            meta_path = os.path.join(papers_dir, "search_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    meta_papers = {p["title"]: p for p in meta.get("papers", [])}
                for p in all_papers:
                    mp = meta_papers.get(p["title"], {})
                    p["abstract"] = mp.get("abstract", "")
                    p["published"] = mp.get("published", "")
                    p["arxiv_id"] = mp.get("arxiv_id", "")
                    p["pdf_url"] = mp.get("pdf_url", "")
                    p["citation_count"] = mp.get("citation_count", 0)
            vector_store.add_papers(all_papers)
            print(f"  📚 已存入本地向量库（共 {vector_store.stats()['total_papers']} 篇）")
        except ImportError:
            pass  # 向量库未安装，跳过
    else:
        print("\n  ❌ 所有 PDF 解析均失败。")



# ==========================================
# 模块 3: Writer（指定 JSON 文件）
# ==========================================
def module_writer():
    """选择 JSON 文件 → 生成 LaTeX 综述"""
    print("\n" + "=" * 55)
    print("  [3] 撰写 LaTeX 综述 (Writer)")
    print("=" * 55)

    latest = find_latest_file("./output_*/all_papers_info.json")
    hint = f"\n  💡 最新文件: {latest}" if latest else ""
    print(hint)

    json_path = input("\n  请输入 JSON 文件路径:\n  > ").strip()
    if not json_path:
        print("  ❌ 路径不能为空！"); return
    if not os.path.exists(json_path):
        print(f"  ❌ 文件不存在: {json_path}"); return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    papers = data.get("papers", [])
    topic = data.get("topic", "")

    if not papers:
        print("  ❌ JSON 中没有论文数据。"); return

    print(f"\n  📂 加载了 {len(papers)} 篇论文信息")
    if topic:
        print(f"  📌 主题: {topic}")

    output_dir = os.path.dirname(json_path)
    safe_topic = sanitize_filename(topic) if topic else "review"
    tex_path = os.path.join(output_dir, f"{safe_topic}_review.tex")

    if os.path.exists(tex_path):
        ow = input("\n  ⚠️ .tex 文件已存在，是否覆盖？(y/n): ").strip().lower()
        if ow != 'y':
            print("  已取消。"); return

    writing_req = input("\n  请输入额外写作要求 (直接回车跳过): ").strip()

    print("\n  -> 正在生成长文综述...")
    latex, whitelist = generate_latex_review(topic, papers, writing_req)
    if not latex:
        print("  ❌ 综述生成失败。"); return

    with open(tex_path, 'w', encoding='utf-8') as f:
        f.write(latex)

    print(f"\n  ✅ 综述已生成！")
    print(f"  📁 {os.path.abspath(tex_path)}")
    print(f"  📄 共 {len(latex)} 字符")


# ==========================================
# 模块 4: Critic（指定 .tex 文件）
# ==========================================
def module_critic():
    """选择 .tex 文件 → 审查引用"""
    print("\n" + "=" * 55)
    print("  [4] 审查引用 (Critic)")
    print("=" * 55)

    latest = find_latest_file("./output_*/*.tex")
    hint = f"\n  💡 最新文件: {latest}" if latest else ""
    print(hint)

    tex_path = input("\n  请输入 .tex 文件路径:\n  > ").strip()
    if not tex_path:
        print("  ❌ 路径不能为空！"); return
    if not os.path.exists(tex_path):
        print(f"  ❌ 文件不存在: {tex_path}"); return

    # 自动匹配同目录下的 JSON
    output_dir = os.path.dirname(tex_path)
    json_path = os.path.join(output_dir, "all_papers_info.json")

    with open(tex_path, 'r', encoding='utf-8') as f:
        latex = f.read()

    if not os.path.exists(json_path):
        print("  ⚠️ 未找到对应的 JSON 文件 (all_papers_info.json)，无法构建白名单。")
        print("  跳过引用审查。")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    whitelist = {}
    for i, paper in enumerate(data.get("papers", []), 1):
        whitelist[f"paper_{i}"] = paper.get("title", "未知")

    print(f"\n  📚 白名单包含 {len(whitelist)} 篇论文")
    print("  -> 正在提取并核对引用...")

    citations = extract_citations(latex)
    valid, hallucinations = verify_citations(citations, whitelist)

    print(f"\n  📊 审查结果:")
    print(f"     合法引用: {len(valid)}")
    print(f"     幻觉引用: {len(hallucinations)}")

    if hallucinations:
        print("\n  ⚠️ 发现以下幻觉引用（不在已下载论文中）:")
        for h in hallucinations:
            print(f"     - {h}")
        print("\n  💡 建议: 返回 Writer 重新生成，或手动修改 .tex 文件。")
    else:
        print("\n  🎉 完美！所有引用均在白名单内，零幻觉！")


# ==========================================
# 模块 5: 全自动管线
# ==========================================
def module_full_pipeline():
    """一次性跑完 Planner → Searcher → Reader → Writer → Critic"""
    print("\n" + "=" * 55)
    print("  [5] 全自动管线（一条龙生成）")
    print("=" * 55)

    topic = input("\n  请输入研究主题 (中文即可):\n  > ").strip()
    if not topic:
        print("  ❌ 主题不能为空！"); return

    safe = sanitize_filename(topic)
    ts = time.strftime("%Y%m%d_%H%M%S")
    download_dir = f"./papers_{safe}_{ts}"
    output_dir = f"./output_{safe}_{ts}"

    # --- Planner + Searcher ---
    print("\n  [1/4] 生成检索策略并下载论文...")
    strategy = generate_search_strategy(topic)
    query = strategy.get('main_query', topic)
    print(f"  检索式: {query}")

    while True:
        try:
            n = int(input("  下载数量: ").strip())
            if n > 0: break
        except: pass

    raw = search_arxiv(query, n)
    if not raw:
        print("  ❌ 未找到论文。"); return
    enriched = enrich_with_citations(raw)
    final = filter_and_rank_papers(enriched, n)
    if not final:
        print("  ❌ 过滤后无论文。"); return

    paths = batch_download(final, download_dir)
    save_search_metadata(final, download_dir, topic)
    if not paths:
        print("  ❌ 下载失败。"); return

    # --- Reader ---
    print(f"\n  [2/4] 解析 {len(paths)} 篇 PDF...")
    mineru_dir = os.path.join(output_dir, "mineru_parsed")
    os.makedirs(mineru_dir, exist_ok=True)
    papers = []
    for pdf_path in paths:
        print(f"  解析 {os.path.basename(pdf_path)}...")
        info = process_single_pdf(pdf_path, mineru_dir)
        if info:
            papers.append(info)

    json_path = os.path.join(output_dir, "all_papers_info.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({"topic": topic, "papers": papers}, f, indent=4, ensure_ascii=False)
    print(f"  ✅ 解析完成: {len(papers)} 篇")

    # --- Writer ---
    if papers:
        print("\n  [3/4] 撰写综述...")
        latex, _ = generate_latex_review(topic, papers, "")
        if latex:
            tex_path = os.path.join(output_dir, f"{safe}_review.tex")
            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex)
            print(f"  ✅ 综述已生成")

    # --- Critic ---
    print("\n  [4/4] 审查引用...")
    if os.path.exists(tex_path) and os.path.exists(json_path):
        with open(tex_path, 'r') as f:
            latex_text = f.read()
        with open(json_path, 'r') as f:
            data = json.load(f)
        whitelist = {f"paper_{i}": p.get("title", "")
                     for i, p in enumerate(data.get("papers", []), 1)}
        cit = extract_citations(latex_text)
        valid, hallucinations = verify_citations(cit, whitelist)
        print(f"  合法: {len(valid)} | 幻觉: {len(hallucinations)}")
        if hallucinations:
            print("  ⚠️ 发现幻觉引用，建议手动修改。")
        else:
            print("  🎉 零幻觉！")

    print("\n" + "█" * 55)
    print("  🎉 全流程完成！")
    print(f"  📁 所有输出: {os.path.abspath(output_dir)}")
    print("█" * 55)


# ==========================================
# 主菜单
# ==========================================
def main():
    while True:
        logger=setup_logger("Main")
        print("\n" + "█" * 55)
        print("  🎓 学术文献综述自动生成工具箱")
        print("█" * 55)
        print("")
        print("  ┌──────────────────────────────────────┐")
        print("  │  1. 🔍 搜索并下载论文                 │")
        print("  │  2. 📖 解析 PDF                      │")
        print("  │  3. ✍️ 撰写综述                       │")
        print("  │  4. ✅ 审查引用                       │")
        print("  │  5. 🤖 全自动管线                     │")
        print("  │     (基础版: Planner→Searcher→       │")
        print("  │       Reader→Writer→Critic)           │")
        print("  │  6. 🧠 Coordinator 智能编排           │")
        print("  │     (带状态看板 + 自动重试 + 容错)      │")
        print("  │  0. ❌ 退出                          │")
        print("  └──────────────────────────────────────┘")

        choice = input("\n  请输入 (0-6): ").strip()

        if choice == "1":
            module_planner_searcher()
        elif choice == "2":
            module_reader()
        elif choice == "3":
            module_writer()
        elif choice == "4":
            module_critic()
        elif choice == "5":
            module_full_pipeline()
        elif choice == "6":                                  # ← 新增
            topic = input("\n  请输入研究主题:\n  > ").strip()
            if topic:
                try:
                    n=int(input("\n  请输入最大论文数 (默认 10): ").strip())
                except ValueError:
                    n=10
                    logger.warning("  无效输入。")
                    continue
                coordinator = Coordinator(topic, n)
                coordinator.run()
        elif choice == "0":
            print("  👋 再见！")
            tracker.print_summary()
            break
        else:
            print("  ⚠️ 无效输入，请输入 0-6。")
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n  👋 已退出。")
        tracker.print_summary()
    except Exception as e:
        print(f"\n  ❌ 异常: {e}")
        import traceback; traceback.print_exc()