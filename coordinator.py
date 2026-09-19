"""
Coordinator Agent - Pipeline 工作流编排器
职责：编排 Planner → Searcher → Reader → Writer → Critic 全流程
架构模式：Pipeline + Conditional Retry + Graceful Degradation

关键设计：
- PipelineState 维护全局状态，各 Agent 读写同一份状态
- Searcher 论文不足 → 用 synonym_query 备选检索式重试
- Reader 部分 PDF 失败 → 跳过不影响后续
- 实时状态看板 → 可视化当前进度
"""

import os
import sys
import time
import json
import glob
import re
from dataclasses import dataclass, field
from typing import Optional
from token_tracker import tracker
from logger import setup_logger

from planner_agent import generate_search_strategy
from searcher_agent import (
    search_arxiv, enrich_with_citations, filter_and_rank_papers,
    batch_download, save_search_metadata, hybrid_search
)
from reader_agent import process_single_pdf
from writer_agent import generate_latex_review
from critic_agent import extract_citations, verify_citations
from checkpoint import save_reader_progress, load_reader_progress, clear_reader_progress
from vector_store import vector_store


# ==========================================
# Pipeline 状态定义
# ==========================================

@dataclass
class PipelineState:
    """Pipeline 全局状态，Coordinator 读写这份状态来编排流程"""
    topic: str = ""
    strategy: dict = field(default_factory=dict)
    papers: list = field(default_factory=list)
    download_dir: str = ""
    parsed_papers: list = field(default_factory=list)
    json_path: str = ""
    latex_content: str = ""
    tex_path: str = ""
    errors: list = field(default_factory=list)

    planner_retries: int = 0
    reader_success: int = 0
    reader_failed: int = 0

    # 各步骤状态
    steps: dict = field(default_factory=lambda: {
        "planner":  "⏳",
        "searcher": "⏳",
        "reader":   "⏳",
        "writer":   "⏳",
        "critic":   "⏳",
    })

    def update_step(self, name: str, status: str):
        self.steps[name] = status

    def status_board(self) -> str:
        """打印实时状态看板"""
        board = f"""
  ┌────────────────────────────────────────┐
  │  📊 Pipeline 状态看板                   │
  │  Topic: {self.topic[:30]:<30s} │
  │                                        │
  │  Planner  → {self.steps['planner']}  Searcher → {self.steps['searcher']}  │
  │  Reader   → {self.steps['reader']}  Writer   → {self.steps['writer']}  │
  │  Critic   → {self.steps['critic']}                        │
  │                                        │
  │  论文: {len(self.papers):<3d}篇  解析: {self.reader_success:<3d}篇  │
  │  错误: {len(self.errors):<3d}个  重试: {self.planner_retries:<3d}次  │
  └────────────────────────────────────────┘"""
        return board


# ==========================================
# Coordinator Agent
# ==========================================

class Coordinator:
    """
    工作流编排 Agent
    Coordinator = State + Steps + Error Handling + Retry Logic
    """

    MAX_PLANNER_RETRIES = 2   # Planner 最大重试次数
    MIN_PAPERS_THRESHOLD = 3   # 最少论文数，低于此值触发备选检索

    def __init__(self, topic: str, max_papers: int = 10):
        self.state = PipelineState(topic=topic)
        self.max_papers = max_papers
        self.logger = setup_logger("Coordinator")
    # ==========================================
    # 步骤 1: Planner
    # ==========================================

    def run_planner(self) -> bool:
        """调用 Planner 生成检索策略，支持重试"""
        self.logger.info(f"\n  [Coordinator] 步骤 1/5: Planner — 生成检索策略")

        for attempt in range(self.MAX_PLANNER_RETRIES + 1):
            try:
                if attempt > 0:
                    self.state.planner_retries += 1
                    print(f"  🔄 第 {attempt} 次重试 Planner...")

                strategy = generate_search_strategy(self.state.topic)
                self.state.strategy = strategy
                self.state.update_step("planner", "✅")
                print(f"  ✅ Planner 成功")
                return True

            except Exception as e:
                self.logger.error(f"  ❌ Planner 失败 (尝试 {attempt+1}/{self.MAX_PLANNER_RETRIES+1}): {e}")
                self.state.errors.append(f"Planner attempt {attempt+1}: {e}")

        self.state.update_step("planner", "❌")
        return False

    # ==========================================
    # 步骤 2: Searcher
    # ==========================================

    def run_searcher(self) -> bool:
        """调用 Searcher 做混合检索，论文不足时触发备用检索式重试"""
        self.logger.info(f"\n  [Coordinator] 步骤 2/5: Searcher — 混合检索")

        query = self.state.strategy.get("exact_query", self.state.topic)
        max_papers = self.max_papers

        try:
            final_papers = hybrid_search(query, self.state.topic, max_papers)

            if not final_papers:
                self.logger.warning("  ⚠️ Searcher 未找到任何论文")
                self.state.update_step("searcher", "⚠️")
                self.state.errors.append("Searcher returned 0 papers")
                return False

            if len(final_papers) < self.MIN_PAPERS_THRESHOLD:
                self.logger.warning(f"  ⚠️ 论文数 {len(final_papers)} 低于阈值 {self.MIN_PAPERS_THRESHOLD}")
                self.state.update_step("searcher", "⚠️")
                self.state.errors.append(f"Only {len(final_papers)} papers found")
                return False

            # 下载论文
            print(f"  📥 下载 {len(final_papers)} 篇论文...")
            safe = re.sub(r'[<>:"/\\|?*]', '', self.state.topic).replace(' ', '_')[:30]
            ts = time.strftime("%Y%m%d_%H%M%S")
            download_dir = f"./papers_{safe}_{ts}"
            paths = batch_download(final_papers, download_dir)
            save_search_metadata(final_papers, download_dir, self.state.topic)

            self.state.papers = final_papers
            self.state.download_dir = download_dir
            self.state.update_step("searcher", "✅")
            print(f"  ✅ Searcher 成功: {len(final_papers)} 篇, 下载 {len(paths)} 篇")
            return True

        except Exception as e:
            self.logger.error(f"  ❌ Searcher 异常: {e}")
            self.state.errors.append(f"Searcher error: {e}")
            self.state.update_step("searcher", "❌")
            return False

    # ==========================================
    # 步骤 2b: 备选检索（Searcher 不足时触发）
    # ==========================================

    def run_alternative_search(self) -> bool:
        """用 synonym_query / related_query 做备选检索"""
        queries_to_try = ["synonym_query", "related_query", "survey_query", "exact_query"]

        for qtype in queries_to_try:
            query = self.state.strategy.get(qtype, "")
            if not query or query == self.state.strategy.get("exact_query", ""):
                continue

            print(f"\n  🔄 尝试备选检索 ({qtype}): {query[:60]}...")
            try:
                alt_papers = hybrid_search(
                    query,
                    self.state.topic,
                    self.max_papers
                )
                if alt_papers and len(alt_papers) >= self.MIN_PAPERS_THRESHOLD:
                    self.state.papers = alt_papers
                    safe = re.sub(r'[<>:"/\\|?*]', '', self.state.topic).replace(' ', '_')[:30]
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    self.state.download_dir = f"./papers_{safe}_{ts}"
                    paths = batch_download(alt_papers, self.state.download_dir)
                    save_search_metadata(alt_papers, self.state.download_dir, self.state.topic)
                    self.state.update_step("searcher", "✅")
                    print(f"  ✅ 备选检索成功 ({qtype}): {len(alt_papers)} 篇")
                    return True
                else:
                    print(f"  ⚠️ 备选检索 ({qtype}) 结果 {len(alt_papers) if alt_papers else 0} 篇，继续尝试下一组")
            except Exception as e:
                self.state.errors.append(f"Alt search ({qtype}) failed: {e}")
                continue

        self.state.update_step("searcher", "⚠️")
        return False

    # ==========================================
    # 步骤 3: Reader
    # ==========================================

    def run_reader(self) -> bool:
        """调用 Reader 解析 PDF，失败部分跳过不影响流程"""
        self.logger.info(f"\n  [Coordinator] 步骤 3/5: Reader — 解析 PDF")

        if not self.state.download_dir:
            self.logger.warning("  ⚠️ 没有下载路径，跳过 Reader")
            self.state.update_step("reader", "⏭️")
            return False

        pdfs = sorted(glob.glob(os.path.join(self.state.download_dir, "*.pdf")))
        if not pdfs:
            self.logger.warning(f"  ⚠️ {self.state.download_dir} 中没有 PDF 文件")
            self.state.update_step("reader", "⚠️")
            return False

        print(f"  📄 共发现 {len(pdfs)} 篇 PDF")

        output_dir = self.state.download_dir.replace("papers_", "output_", 1)
        mineru_dir = os.path.join(output_dir, "mineru_parsed")
        os.makedirs(mineru_dir, exist_ok=True)
        json_path = os.path.join(output_dir, "all_papers_info.json")

        # 加载已有进度
        all_papers = []
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                all_papers = json.load(f).get("papers", [])

        completed = set(load_reader_progress(output_dir) or [])
        pending = [p for p in pdfs if os.path.basename(p) not in completed]

        if not pending:
            print(f"  ✅ 所有 PDF 均已解析过")
            self.state.parsed_papers = all_papers
            self.state.json_path = json_path
            self.state.reader_success = len(all_papers)
            self.state.update_step("reader", "✅")
            return True

        print(f"  📖 待解析 {len(pending)} 篇（已有 {len(all_papers)} 篇缓存）")

        success_count = 0
        fail_count = 0

        for idx, pdf_path in enumerate(pending, 1):
            pdf_name = os.path.basename(pdf_path)
            print(f"  --- [{idx}/{len(pending)}] {pdf_name} ---")

            try:
                info = process_single_pdf(pdf_path, mineru_dir)
                if info:
                    all_papers.append(info)
                    completed.add(pdf_name)
                    save_reader_progress(output_dir, list(completed))
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump({"papers": all_papers, "topic": self.state.topic},
                                  f, indent=4, ensure_ascii=False)
                    success_count += 1
                    print(f"    ✅ 进度: {len(completed)}/{len(pdfs)}")
                else:
                    fail_count += 1
                    print(f"    ⚠️ 解析失败（跳过）")
            except Exception as e:
                fail_count += 1
                self.state.errors.append(f"Reader failed on {pdf_name}: {e}")
                print(f"    ❌ 异常: {e}（跳过）")

        self.state.parsed_papers = all_papers
        self.state.json_path = json_path
        self.state.reader_success = success_count
        self.state.reader_failed = fail_count

        # 存入本地向量库
        self._save_to_vector_store(all_papers)

        if success_count > 0:
            self.state.update_step("reader", "✅")
        elif fail_count > 0 and success_count == 0:
            self.state.update_step("reader", "❌")
        else:
            self.state.update_step("reader", "⚠️")

        if os.path.exists(json_path):
            clear_reader_progress(output_dir)

        return success_count > 0

    def _save_to_vector_store(self, papers: list):
        """将解析结果存入本地向量库"""
        try:
            meta_path = os.path.join(self.state.download_dir, "search_metadata.json")
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                    meta_papers = {p["title"]: p for p in meta.get("papers", [])}
                for p in papers:
                    mp = meta_papers.get(p["title"], {})
                    p["abstract"] = mp.get("abstract", "")
                    p["published"] = mp.get("published", "")
                    p["arxiv_id"] = mp.get("arxiv_id", "")
                    p["pdf_url"] = mp.get("pdf_url", "")
                    p["citation_count"] = mp.get("citation_count", 0)
            vector_store.add_papers(papers)
            print(f"  📚 已存入本地向量库（共 {vector_store.stats()['total_papers']} 篇）")
        except Exception as e:
            self.state.errors.append(f"Vector store save failed: {e}")
            pass

    # ==========================================
    # 步骤 4: Writer
    # ==========================================

    def run_writer(self) -> bool:
        """调用 Writer 生成 LaTeX 综述"""
        self.logger.info(f"\n  [Coordinator] 步骤 4/5: Writer — 撰写综述")

        if not self.state.parsed_papers:
            self.logger.warning("  ⚠️ 没有解析好的论文，尝试从 JSON 加载")
            if self.state.json_path and os.path.exists(self.state.json_path):
                with open(self.state.json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.state.parsed_papers = data.get("papers", [])

        if not self.state.parsed_papers:
            self.logger.error("  ❌ 没有论文数据，无法撰写")
            self.state.update_step("writer", "❌")
            return False

        print(f"  ✍️ 基于 {len(self.state.parsed_papers)} 篇论文生成综述...")

        try:
            latex, whitelist = generate_latex_review(
                topic=self.state.topic,
                papers_info=self.state.parsed_papers,
                writing_req=""
            )

            if not latex:
                self.logger.error("  ❌ Writer 返回空")
                self.state.update_step("writer", "❌")
                return False

            safe = re.sub(r'[<>:"/\\|?*]', '', self.state.topic).replace(' ', '_')[:30]
            output_dir = self.state.download_dir.replace("papers_", "output_", 1)
            tex_path = os.path.join(output_dir, f"{safe}_review.tex")

            with open(tex_path, 'w', encoding='utf-8') as f:
                f.write(latex)

            self.state.latex_content = latex
            self.state.tex_path = tex_path
            self.state.update_step("writer", "✅")
            print(f"  ✅ Writer 成功: {len(latex)} 字符 → {tex_path}")
            return True

        except Exception as e:
            self.logger.error(f"  ❌ Writer 异常: {e}")
            self.state.errors.append(f"Writer error: {e}")
            self.state.update_step("writer", "❌")
            return False

    # ==========================================
    # 步骤 5: Critic
    # ==========================================

    def run_critic(self) -> bool:
        """调用 Critic 审查引用"""
        self.logger.info(f"\n  [Coordinator] 步骤 5/5: Critic — 审查引用")

        if not self.state.tex_path or not os.path.exists(self.state.tex_path):
            self.logger.error("  ❌ 找不到 .tex 文件")
            self.state.update_step("critic", "❌")
            return False

        # 找对应的 JSON（同目录下的 all_papers_info.json）
        output_dir = os.path.dirname(self.state.tex_path)
        json_path = os.path.join(output_dir, "all_papers_info.json")

        if not os.path.exists(json_path):
            self.logger.warning("  ⚠️ 找不到 all_papers_info.json，用 state 中的论文数据")
            if not self.state.parsed_papers:
                self.state.update_step("critic", "⚠️")
                return False
            whitelist = {f"paper_{i}": p.get("title", "") for i, p in enumerate(self.state.parsed_papers, 1)}
        else:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            whitelist = {f"paper_{i}": p.get("title", "") for i, p in enumerate(data.get("papers", []), 1)}

        try:
            with open(self.state.tex_path, 'r', encoding='utf-8') as f:
                latex = f.read()

            citations = extract_citations(latex)
            valid, hallucinations = verify_citations(citations, whitelist)

            print(f"  📊 审查结果:")
            print(f"     合法引用: {len(valid)}")
            print(f"     幻觉引用: {len(hallucinations)}")

            if hallucinations:
                print(f"  ⚠️ 发现 {len(hallucinations)} 个幻觉引用:")
                for h in hallucinations[:5]:
                    print(f"     - {h}")
                self.state.update_step("critic", "⚠️")
                self.state.errors.append(f"{len(hallucinations)} hallucinated citations")
            else:
                print(f"  🎉 零幻觉！")
                self.state.update_step("critic", "✅")

            return True

        except Exception as e:
            self.logger.error(f"  ❌ Critic 异常: {e}")
            self.state.errors.append(f"Critic error: {e}")
            self.state.update_step("critic", "❌")
            return False

    # ==========================================
    # 主编排逻辑
    # ==========================================

    def run(self):
        """执行完整的 Pipeline 编排"""
        print("\n" + "█" * 55)
        print("  🧠 Coordinator Agent — 全自动管线编排")
        print("  架构模式: Pipeline + Conditional Retry + Graceful Degradation")
        print("█" * 55)
        print(f"\n  🎯 主题: {self.state.topic}")

        # --- Step 1: Planner ---
        planner_ok = self.run_planner()
        print(self.state.status_board())

        if not planner_ok:
            print("\n  ❌ Planner 失败，无法继续")
            self._print_summary()
            return

        # --- Step 2: Searcher + 备选检索 ---
        searcher_ok = self.run_searcher()

        # 如果精确匹配结果不足，尝试备选检索式
        if not searcher_ok:
            print(f"\n  🔄 精确检索结果不足，尝试备选检索式...")
            searcher_ok = self.run_alternative_search()

        print(self.state.status_board())

        if not searcher_ok:
            print("\n  ❌ Searcher 失败，无法继续")
            self._print_summary()
            return

        # --- Step 3: Reader (失败部分跳过) ---
        self.run_reader()
        print(self.state.status_board())

        # --- Step 4: Writer ---
        self.run_writer()
        print(self.state.status_board())

        # --- Step 5: Critic ---
        self.run_critic()
        print(self.state.status_board())

        # --- 最终报告 ---
        self._print_summary()

    # ==========================================
    # 最终报告
    # ==========================================

    def _print_summary(self):
        """打印 Pipeline 最终执行报告"""
        print("\n" + "█" * 55)
        print("  📋 Pipeline 执行报告")
        print("█" * 55)

        print(f"\n  🎯 主题: {self.state.topic}")
        print()
        print(f"  Planner  → {self.state.steps['planner']}")
        print(f"  Searcher → {self.state.steps['searcher']}")
        print(f"  Reader   → {self.state.steps['reader']}  ({self.state.reader_success} 成功, {self.state.reader_failed} 失败)")
        print(f"  Writer   → {self.state.steps['writer']}")
        print(f"  Critic   → {self.state.steps['critic']}")
        print()

        all_ok = all(s == "✅" for s in self.state.steps.values() if s != "⏭️")
        if all_ok:
            print(f"  🎉 全部步骤成功完成！")
        else:
            failed = [k for k, v in self.state.steps.items() if v in ("❌", "⚠️")]
            print(f"  ⚠️ 部分步骤存在问题: {', '.join(failed)}")

        if self.state.errors:
            print(f"\n  📝 错误日志 ({len(self.state.errors)} 条):")
            for e in self.state.errors[-5:]:
                print(f"     • {e}")

        if self.state.tex_path:
            print(f"\n  📁 输出文件:")
            print(f"     • {self.state.tex_path}")
            if self.state.json_path:
                print(f"     • {self.state.json_path}")

        print(f"\n  💰 Token 统计:")
        tracker.print_summary()
        print()


# ==========================================
# 独立测试入口
# ==========================================

if __name__ == "__main__":
    topic = input("  请输入研究主题: ").strip()
    if topic:
        coordinator = Coordinator(topic)
        coordinator.run()
    else:
        print("  ❌ 主题不能为空")