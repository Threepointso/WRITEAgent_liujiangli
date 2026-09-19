import os
import re
import json
import glob

# ==========================================
# 1. 智能查找函数
# ==========================================

def find_latest_tex() -> str:
    """查找最新的 review_output.tex"""
    files = glob.glob("./review_output*.tex")
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def find_latest_json(tex_path: str) -> str:
    """查找修改时间早于 TEX 文件的最新 all_papers.json"""
    tex_mtime = os.path.getmtime(tex_path)
    
    # 查找所有 output 目录下的 all_papers.json
    pattern = "./output_*/**/*_all_papers.json"
    files = glob.glob(pattern, recursive=True)
    
    # 过滤出在 TEX 生成之前修改过的 JSON
    valid_files = [f for f in files if os.path.getmtime(f) <= tex_mtime]
    
    if not valid_files:
        # 如果找不到，就返回所有 JSON 中最新的一个
        return max(files, key=os.path.getmtime) if files else None
        
    return max(valid_files, key=os.path.getmtime)


# ==========================================
# 2. 核心校验逻辑
# ==========================================

def extract_citations(latex_content: str) -> list:
    pattern = r'\\cite\{([^}]+)\}'
    matches = re.findall(pattern, latex_content)
    citations = []
    for match in matches:
        for cite_id in match.split(','):
            citations.append(cite_id.strip())
    return list(set(citations))

def verify_citations(citations: list, whitelist: dict):
    print("\n️ [Critic] 正在核对引用白名单...")
    valid_cites = []
    hallucination_cites = []
    
    for cite in citations:
        if cite in whitelist:
            valid_cites.append(cite)
        else:
            hallucination_cites.append(cite)
            
    return valid_cites, hallucination_cites


# ==========================================
# 3. 主程序入口
# ==========================================

def main():
    print("\n" + "="*60)
    print("🕵️ Critic Agent - 零幻觉交叉验证模块")
    print("="*60)
    
    # 1. 自动查找 TEX 文件
    tex_path = find_latest_tex()
    if not tex_path:
        print("❌ 找不到 review_output.tex！请先运行 Writer Agent。")
        return
    print(f"\n📄 找到 LaTeX 文件: {os.path.abspath(tex_path)}")
    
    # 2. 自动查找对应的 JSON 白名单
    json_path = find_latest_json(tex_path)
    if not json_path:
        print("❌ 找不到对应的 all_papers.json！")
        return
    print(f"📂 找到文献库: {os.path.abspath(json_path)}")
    
    # 3. 加载白名单
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    whitelist = {}
    for i, paper in enumerate(data.get("papers", []), 1):
        whitelist[f"paper_{i}"] = paper.get("title", "未知")
        
    print(f"✅ 白名单包含 {len(whitelist)} 篇合法文献。")
    
    # 4. 读取 LaTeX 并提取引用
    with open(tex_path, 'r', encoding='utf-8') as f:
        latex_content = f.read()
        
    citations = extract_citations(latex_content)
    print(f"\n🔍 在 LaTeX 中共提取到 {len(citations)} 个独立引用标签: {citations}")
    
    # 5. 交叉验证
    valid, hallucinations = verify_citations(citations, whitelist)
    
    # 6. 输出审计报告
    print("\n" + "-"*40)
    print("📊 【Critic 审计报告】")
    print("-"*40)
    print(f"✅ 合法引用: {len(valid)} 个")
    print(f"❌ 幻觉引用: {len(hallucinations)} 个")
    
    if hallucinations:
        print("\n⚠️ 警告！发现以下捏造的引用标签 (不在白名单中):")
        for h in hallucinations:
            print(f"   - \\cite{{{h}}}")
    else:
        print("\n🎉 完美！所有引用均来自真实文献库，实现 100% 零幻觉！")
        
    print("-"*40 + "\n")


if __name__ == "__main__":
    main()