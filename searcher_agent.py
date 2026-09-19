print("🚀 探针：代码已经开始执行啦！")
"""
Searcher Agent - 学术文献自动检索与下载模块
功能：根据关键词自动检索 arXiv 论文，通过 Semantic Scholar 过滤高质量文献，并发下载 PDF
"""

import os
import re
import time
import arxiv
import requests
from tqdm import tqdm
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from semanticscholar import SemanticScholar
from vector_store import vector_store

# ==========================================
# 1. 配置区域 (根据需求修改)
# ==========================================

# 下载并发数（arXiv 对并发敏感，建议不超过 3）
MAX_DOWNLOAD_WORKERS = 3

# Semantic Scholar API 请求间隔（秒）
S2_API_INTERVAL = 1.5

# arXiv API 请求间隔（秒）
ARXIV_API_INTERVAL = 3.0

# 默认下载的论文数量
DEFAULT_MAX_PAPERS = 10

# 论文最低引用量阈值（0 = 不过滤，建议新项目设为 0）
MIN_CITATION_COUNT = 0

# PDF 下载超时时间（秒）
DOWNLOAD_TIMEOUT = 60

# 每个 PDF 下载失败后的重试次数
MAX_RETRY = 2


# ==========================================
# 2. 工具函数
# ==========================================

def sanitize_filename(name: str, max_length: int = 80) -> str:
    """
    清理字符串，使其成为合法的 Windows 文件名
    """
    illegal_chars = r'[<>:"/\\|?*]'
    cleaned = re.sub(illegal_chars, '', name)
    cleaned = re.sub(r'\s+', '_', cleaned.strip())
    return cleaned[:max_length]

def fix_arxiv_query(query: str) -> str:
    """
    【核心新增】自动修正 arXiv 检索式，防止因缺少字段前缀导致 HTTP 500 错误。
    作为防御性编程的兜底逻辑，确保即使 LLM 输出不规范，代码也能自动修复。
    """
    valid_prefixes = ['all:', 'ti:', 'au:', 'ab:', 'co:', 'cat:', 'id:']
    # 检查是否已经包含了合法的字段前缀（忽略大小写）
    has_prefix = any(prefix in query.lower() for prefix in valid_prefixes)
    
    if not has_prefix:
        # 如果没有前缀，尝试在双引号前加上 all:
        # 例如: "Mixture of Experts" OR "MoE" -> all:"Mixture of Experts" OR all:"MoE"
        fixed_query = re.sub(r'(?<!all:)"', 'all:"', query, flags=re.IGNORECASE)
        
        # 如果连引号都没有（比如纯单词 AND 纯单词），直接整体包裹 all:()
        if fixed_query == query and '"' not in query:
            fixed_query = f'all:({query})'
            
        print(f"   🛠️ [自动修正] 检索式缺少字段前缀，已自动补充 -> {fixed_query}")
        return fixed_query
    return query


# ==========================================
# 3. 核心功能函数
# ==========================================

def search_arxiv(query: str, max_results: int) -> list:
    """
    调用 arXiv API 检索论文
    """
    # 【核心修改】：在发起请求前，自动修正检索式语法
    query = fix_arxiv_query(query)
    
    print(f"\n🔍 [arXiv] 正在检索: '{query}'")
    print(f"   请求数量: {max_results} 篇 (会多搜用于后续过滤)...")

    client = arxiv.Client(
        page_size=50,       
        delay_seconds=3.0,  
        num_retries=3       # arXiv 库自带的重试机制
    )

    search = arxiv.Search(
        query=query,
        max_results=max_results * 2,  
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        for result in client.results(search):
            paper = {
                "title": result.title.replace('\n', ' ').strip(),
                "authors": [author.name for author in result.authors],
                "abstract": result.summary.replace('\n', ' ').strip(),
                "published": result.published.strftime("%Y-%m-%d"),
                "arxiv_id": result.entry_id.split("/abs/")[-1],
                "pdf_url": result.pdf_url,
                "citation_count": None  
            }
            papers.append(paper)
    except Exception as e:
        print(f"   ⚠️ arXiv 检索异常: {e}")

    print(f"   ✅ arXiv 返回 {len(papers)} 篇候选论文")
    return papers


def enrich_with_citations(papers: list) -> list:
    """
    调用 Semantic Scholar API 获取每篇论文的引用量
    """
    if not papers:
        return papers

    print(f"\n📊 [Semantic Scholar] 正在查询 {len(papers)} 篇论文的引用量...")
    sch = SemanticScholar()

    enriched = []
    for i, paper in enumerate(papers):
        arxiv_id = paper["arxiv_id"]
        short_title = paper["title"][:50] + "..." if len(paper["title"]) > 50 else paper["title"]

        try:
            ss_paper = sch.paper(f"arXiv:{arxiv_id}", fields=["citationCount", "year"])

            if ss_paper:
                paper["citation_count"] = ss_paper.get("citationCount", 0) or 0
                if ss_paper.get("year"):
                    paper["year"] = str(ss_paper["year"])
            else:
                paper["citation_count"] = 0

            status = f"引用量: {paper['citation_count']}"

        except Exception as e:
            paper["citation_count"] = 0
            status = f"查询失败 (默认0)"

        print(f"   [{i+1}/{len(papers)}] {short_title} → {status}")
        enriched.append(paper)

        if i < len(papers) - 1:
            time.sleep(S2_API_INTERVAL)

    return enriched


def filter_and_rank_papers(papers: list, max_papers: int) -> list:
    """
    根据引用量过滤和排序论文
    """
    print(f"\n📋 [过滤] 最低引用量阈值: {MIN_CITATION_COUNT}")

    filtered = [p for p in papers if (p.get("citation_count") or 0) >= MIN_CITATION_COUNT]
    filtered.sort(key=lambda x: x.get("citation_count", 0), reverse=True)
    final = filtered[:max_papers]

    print(f"   过滤后剩余: {len(filtered)} 篇")
    print(f"   最终选取: {len(final)} 篇\n")

    print("   " + "-" * 70)
    for i, p in enumerate(final, 1):
        title = p["title"][:55] + "..." if len(p["title"]) > 55 else p["title"]
        citations = p.get("citation_count", "?")
        date = p.get("published", "?")[:10]
        print(f"   {i:2d}. [{citations:>4} 引用] ({date}) {title}")
    print("   " + "-" * 70)

    return final


def download_single_pdf(paper: dict, download_dir: str) -> dict:
    """
    下载单篇论文的 PDF
    """
    safe_title = sanitize_filename(paper["title"])
    arxiv_id_safe = paper["arxiv_id"].replace("/", "_").replace(".", "_")
    filename = f"{safe_title}_{arxiv_id_safe}.pdf"
    filepath = os.path.join(download_dir, filename)

    result = {
        "success": False,
        "path": filepath,
        "title": paper["title"],
        "arxiv_id": paper["arxiv_id"]
    }

    if os.path.exists(filepath) and os.path.getsize(filepath) > 10240:
        result["success"] = True
        result["skipped"] = True
        return result

    pdf_url = paper.get("pdf_url", "")
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{paper['arxiv_id']}.pdf"

    for attempt in range(1, MAX_RETRY + 1):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AcademicBot/1.0'
            }
            response = requests.get(
                pdf_url,
                headers=headers,
                stream=True,
                timeout=DOWNLOAD_TIMEOUT
            )
            response.raise_for_status()

            total_size = int(response.headers.get('content-length', 0))
            with open(filepath, 'wb') as f:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

            if os.path.getsize(filepath) > 10240:
                result["success"] = True
                return result
            else:
                os.remove(filepath)
                if attempt < MAX_RETRY:
                    time.sleep(2)
                    continue

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRY:
                time.sleep(2 * attempt)  
                continue
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                except OSError:
                    pass

    return result


def batch_download(papers: list, download_dir: str) -> list:
    """
    使用线程池并发下载多篇论文的 PDF
    """
    print(f"\n⬇️  [下载] 开始下载 {len(papers)} 篇 PDF")
    print(f"   目标目录: {os.path.abspath(download_dir)}")
    print(f"   并发数: {MAX_DOWNLOAD_WORKERS}\n")

    os.makedirs(download_dir, exist_ok=True)

    success_paths = []
    fail_titles = []

    with ThreadPoolExecutor(max_workers=MAX_DOWNLOAD_WORKERS) as executor:
        future_to_paper = {
            executor.submit(download_single_pdf, paper, download_dir): paper
            for paper in papers
        }

        for future in tqdm(as_completed(future_to_paper), total=len(papers), desc="   下载进度"):
            try:
                result = future.result()
                if result["success"]:
                    success_paths.append(result["path"])
                    skipped = " (已缓存)" if result.get("skipped") else ""
                    short = result["title"][:40] + "..." if len(result["title"]) > 40 else result["title"]
                    tqdm.write(f"   ✅ {short}{skipped}")
                else:
                    fail_titles.append(result["title"][:40])
                    tqdm.write(f"   ❌ 下载失败: {result['title'][:40]}...")
            except Exception as e:
                tqdm.write(f"   ❌ 异常: {e}")

    print(f"\n📦 [下载完成]")
    print(f"   成功: {len(success_paths)}/{len(papers)}")
    print(f"   失败: {len(fail_titles)}/{len(papers)}")

    return success_paths
def hybrid_search(query: str, topic: str, max_papers: int = 10) -> list:
    """
    混合检索：arXiv 关键词 + 本地向量库语义检索
    先用 arXiv 搜，再查本地向量库，两路结果合并去重
    """
    # 路 1: arXiv 检索
    print(f"\n  -> 正在检索 arXiv...")
    raw = search_arxiv(query, max_papers)
    if not raw:
        print("  ⚠️ arXiv 未找到结果")
        return []

    enriched = enrich_with_citations(raw)
    final = filter_and_rank_papers(enriched, max_papers)

    # 路 2: 本地向量库检索（补充已有论文）
    try:
        local_hits = vector_store.search(topic, k=5)
        if local_hits:
            print(f"\n  📚 [本地向量库] 命中 {len(local_hits)} 篇历史论文：")
            for h in local_hits:
                sim = h.get('similarity', 0)
                print(f"      [{sim:.0%}] {h['title'][:60]}")

            # 去重合并（按标题去重）
            existing_titles = {p["title"] for p in final}
            for h in local_hits:
                if h["title"] not in existing_titles:
                    final.append({
                        "title": h["title"],
                        "authors": h.get("authors", []),
                        "abstract": h.get("abstract", ""),
                        "published": h.get("published", ""),
                        "arxiv_id": h.get("arxiv_id", ""),
                        "pdf_url": h.get("pdf_url", ""),
                        "citation_count": h.get("citation_count", 0),
                        "_from_local_db": True
                    })
            print(f"\n  ✅ 合并后共 {len(final)} 篇（向量库贡献 {len(local_hits)} 篇）")
    except ImportError:
        pass  # 向量库未安装，跳过

    return final

def save_search_metadata(papers: list, download_dir: str, query: str):
    """
    保存检索元数据为 JSON 文件
    """
    import json

    metadata = {
        "query": query,
        "search_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_papers": len(papers),
        "papers": [
            {
                "title": p["title"],
                "authors": p["authors"],
                "published": p.get("published", "未知"),
                "arxiv_id": p["arxiv_id"],
                "citation_count": p.get("citation_count", 0),
                "pdf_url": p.get("pdf_url", ""),
                "abstract": p.get("abstract", "")[:500] 
            }
            for p in papers
        ]
    }

    metadata_path = os.path.join(download_dir, "search_metadata.json")
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"   📄 检索元数据已保存: {metadata_path}")


# ==========================================
# 4. 主程序入口
# ==========================================

def main():
    print("\n" + "=" * 70)
    print("🔍 Searcher Agent - 学术文献自动检索与下载模块")
    print("=" * 70)

    print("\n[步骤1] 请输入检索关键词")
    print("  提示: 支持 arXiv 语法，如 'all:LLM AND all:agent', 'ti:document parsing'")
    query = input("  关键词 > ").strip()
    if not query:
        query = "all:Large Language Model"
        print(f"  未输入，使用默认: {query}")

    try:
        max_papers = int(input(f"\n[步骤2] 需要下载多少篇论文？(默认 {DEFAULT_MAX_PAPERS}): ").strip() or DEFAULT_MAX_PAPERS)
    except ValueError:
        max_papers = DEFAULT_MAX_PAPERS

    safe_query = sanitize_filename(query)[:30]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    download_dir = f"./papers_{safe_query}_{timestamp}"

    raw_papers = search_arxiv(query, max_papers)
    if not raw_papers:
        print("\n❌ 未找到任何论文，请尝试更换关键词。")
        return

    enriched_papers = enrich_with_citations(raw_papers)
    final_papers = filter_and_rank_papers(enriched_papers, max_papers)
    if not final_papers:
        print("\n❌ 过滤后没有符合条件的论文，请降低引用量阈值。")
        return

    confirm = input(f"\n[确认] 准备下载 {len(final_papers)} 篇论文到:\n  {os.path.abspath(download_dir)}\n  是否继续？(y/n): ").strip().lower()

    if confirm != 'y':
        print("已取消。")
        return

    success_paths = batch_download(final_papers, download_dir)
    save_search_metadata(final_papers, download_dir, query)

    print("\n" + "=" * 70)
    print("🎉 Searcher Agent 任务完成！")
    print("=" * 70)
    print(f"\n📁 PDF 保存位置: {os.path.abspath(download_dir)}")
    print(f"📄 成功下载: {len(success_paths)} 篇\n")
    print("💡 下一步：将上述目录路径输入给 Reader Agent 进行批量解析！")
    print(f"   运行命令: python reader_agent.py")
    print(f"   输入路径: {os.path.abspath(download_dir)}")
    print("=" * 70 + "\n")

    return download_dir


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断程序。已下载的文件保留在本地。")
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        import traceback
        traceback.print_exc()