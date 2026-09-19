"""
Critic Agent 单元测试
测试核心逻辑：引用提取 + 白名单校验
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from critic_agent import extract_citations, verify_citations


# ============================================
# 测试 extract_citations
# ============================================

def test_extract_single_citation():
    latex = "相关工作见 \\cite{paper_1}。"
    result = extract_citations(latex)
    assert result == ["paper_1"], f"期望 ['paper_1']，实际得到 {result}"
    print("  ✅ test_extract_single_citation 通过")


def test_extract_multiple_citations():
    latex = "方法A\\cite{paper_1}和方法B\\cite{paper_2}各有优势。"
    result = extract_citations(latex)
    assert set(result) == {"paper_1", "paper_2"}, f"期望两个 paper_ID，实际得到 {result}"
    print("  ✅ test_extract_multiple_citations 通过")


def test_extract_comma_separated():
    latex = "多项研究\\cite{paper_1,paper_2,paper_3}表明..."
    result = extract_citations(latex)
    assert set(result) == {"paper_1", "paper_2", "paper_3"}, f"期望三个 paper_ID，实际得到 {result}"
    print("  ✅ test_extract_comma_separated 通过")


def test_extract_no_citation():
    latex = "这篇综述没有引用任何文献。"
    result = extract_citations(latex)
    assert result == [], f"期望空列表，实际得到 {result}"
    print("  ✅ test_extract_no_citation 通过")


# ============================================
# 测试 verify_citations
# ============================================

def test_verify_all_valid():
    citations = ["paper_1", "paper_2"]
    whitelist = {"paper_1": "论文A", "paper_2": "论文B", "paper_3": "论文C"}
    valid, hallucinations = verify_citations(citations, whitelist)
    assert len(valid) == 2, f"期望 2 个合法，实际 {len(valid)}"
    assert len(hallucinations) == 0, f"期望 0 个幻觉，实际 {len(hallucinations)}"
    print("  ✅ test_verify_all_valid 通过")


def test_verify_with_hallucinations():
    citations = ["paper_1", "paper_999"]
    whitelist = {"paper_1": "论文A", "paper_2": "论文B"}
    valid, hallucinations = verify_citations(citations, whitelist)
    assert len(valid) == 1, f"期望 1 个合法，实际 {len(valid)}"
    assert len(hallucinations) == 1, f"期望 1 个幻觉，实际 {len(hallucinations)}"
    assert "paper_999" in hallucinations, f"期望 paper_999 在幻觉中"
    print("  ✅ test_verify_with_hallucinations 通过")


def test_verify_empty_citations():
    citations = []
    whitelist = {"paper_1": "论文A"}
    valid, hallucinations = verify_citations(citations, whitelist)
    assert len(valid) == 0
    assert len(hallucinations) == 0
    print("  ✅ test_verify_empty_citations 通过")


def test_verify_all_hallucinations():
    citations = ["fake_1", "fake_2"]
    whitelist = {"paper_1": "论文A"}
    valid, hallucinations = verify_citations(citations, whitelist)
    assert len(valid) == 0
    assert len(hallucinations) == 2
    print("  ✅ test_verify_all_hallucinations 通过")


# ============================================
# 运行所有测试
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("  🧪 Critic Agent 单元测试")
    print("=" * 50)
    
    tests = [
        test_extract_single_citation,
        test_extract_multiple_citations,
        test_extract_comma_separated,
        test_extract_no_citation,
        test_verify_all_valid,
        test_verify_with_hallucinations,
        test_verify_empty_citations,
        test_verify_all_hallucinations,
    ]
    
    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  ❌ {test.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"  ❌ {test.__name__} 抛出异常: {e}")
            failed += 1
    
    print(f"\n{'=' * 50}")
    print(f"  结果: {passed}/{len(tests)} 通过, {failed} 失败")
    print(f"{'=' * 50}\n")