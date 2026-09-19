"""
Planner Agent Query 改写质量评估
评估方法：人工标注标准答案 → 调用 Planner → LLM 对比打分
语法检查：硬编码正则（不依赖 LLM）
用法：python evaluate.py
"""

import json
import re
from planner_agent import generate_search_strategy
from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from logger import setup_logger

# ==========================================
# 配置
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

settings = AppSettings()

# ==========================================
# 你手写的标准答案（这是你的人工标注）
# ==========================================
TEST_CASES = [
    {
        "topic": "MoE 在大型语言模型中的应用",
        "ideal": {
            "exact_query": 'all:"Mixture of Experts" AND all:"Large Language Model"',
            "synonym_query": 'all:(MoE OR "expert routing") AND all:(LLM OR "large language model")',
            "related_query": 'all:("sparse gating" OR "conditional computation" OR "load balancing") AND all:transformer',
            "survey_query": 'all:(MoE OR "mixture of experts") AND all:(survey OR review OR comprehensive)',
        },
        "comment": "MoE 在 LLM 中的使用，精确=核心词，同义=缩写，相关=路由/负载均衡"
    },
    {
        "topic": "大模型参数高效微调",
        "ideal": {
            "exact_query": 'all:("parameter efficient" OR "fine tuning") AND all:"large language model"',
            "synonym_query": 'all:(PEFT OR adapter OR LoRA)',
            "related_query": 'all:(prefix tuning OR prompt tuning OR soft prompt)',
            "survey_query": 'all:(PEFT OR "parameter efficient") AND all:(survey OR review)',
        },
        "comment": "PEFT 方向，精确=全称，同义=缩写/LoRA，相关=前缀微调"
    },
    {
        "topic": "Transformer 推理加速",
        "ideal": {
            "exact_query": 'all:"Transformer" AND all:("inference acceleration" OR "inference speedup")',
            "synonym_query": 'all:(KV cache OR "attention acceleration") AND all:transformer',
            "related_query": 'all:(quantization OR pruning OR distillation) AND all:transformer',
            "survey_query": 'all:(Transformer OR LLM) AND all:("inference acceleration" OR "efficient inference") AND all:(survey OR review)',
        },
        "comment": "推理加速技术栈，精确=核心词，相关=量化/剪枝/蒸馏"
    },
]

# ==========================================
# 工具函数
# ==========================================

def extract_json_from_text(text: str) -> dict:
    """鲁棒的 JSON 提取，LLM 输出不稳定时兜底"""
    # 1. 直接解析
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. 提取 ```json ... ``` 代码块
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # 3. 提取第一个 { 到最后一个 } 之间的内容
    match = re.search(r'\{.*\}', text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    # 4. 全部失败，抛异常
    raise ValueError(f"无法从 LLM 输出中提取 JSON:\n{text[:200]}")


def check_syntax_hardcoded(query: str) -> int:
    """硬编码检查检索式语法，比 LLM 判断更准确"""
    if not query or not query.strip():
        return 1

    # 检查是否包含字段前缀
    has_prefix = bool(re.search(r'\b(all|ti|au|abs):', query))
    # 检查括号是否匹配
    parens_balanced = query.count('(') == query.count(')')
    # 检查引号是否匹配
    quotes_balanced = query.count('"') % 2 == 0

    if has_prefix and parens_balanced and quotes_balanced:
        return 5
    elif has_prefix and (parens_balanced or quotes_balanced):
        return 3
    elif has_prefix:
        return 2
    else:
        return 1


# ==========================================
# 评估函数
# ==========================================

def evaluate_single_case(topic: str, ideal: dict, comment: str) -> dict:
    """评估单个 Topic 的 Planner 输出质量"""
    logger = setup_logger("Evaluate")

    print(f"\n  ─── 正在评估: {topic} ───")

    # 1. 调用 Planner 生成检索式
    try:
        strategy = generate_search_strategy(topic)
    except Exception as e:
        print(f"  ❌ Planner 调用失败: {e}")
        return {"topic": topic, "total_score": 0, "error": str(e)}

    # 提取实际输出的 4 个 query
    actual = {k: strategy.get(k, "") for k in ["exact_query", "synonym_query", "related_query", "survey_query"]}

    # 2. 硬编码检查语法（不依赖 LLM）
    syntax_scores = {k: check_syntax_hardcoded(v) for k, v in actual.items()}
    avg_syntax = sum(syntax_scores.values()) / len(syntax_scores)
    syntax_detail = f"exact={syntax_scores['exact_query']} synonym={syntax_scores['synonym_query']} related={syntax_scores['related_query']} survey={syntax_scores['survey_query']}"

    # 3. LLM 评估语义维度（强调语义等效，避免字面匹配扣分）
    eval_prompt = f"""
你是一位严格的学术检索评估专家。对比「标准答案」和「AI 实际输出」，评估 AI 的 Query 改写质量。

## 研究主题
{topic}

## 标准答案（人工标注的参考基准）
{json.dumps(ideal, ensure_ascii=False, indent=2)}

## AI 实际输出（Planner Agent 生成）
{json.dumps(actual, ensure_ascii=False, indent=2)}

## ⚠️ 核心评估原则
1. **语义等效优先**：检索式不要求字面完全一致。只要 AI 输出的词汇在学术语义上覆盖了标准答案的核心概念，即可给高分。
2. **关注概念遗漏**：重点检查 AI 是否漏掉了标准答案中的关键子方向（如 MoE 漏掉了"路由/负载均衡"）。

## 评分维度（每个维度 1~5 分）
1. **exact_query（精确匹配）**：是否准确抓住了最核心的术语全称/缩写？
2. **synonym_query（同义扩展）**：是否合理覆盖了该主题下常见的同义词、别名、不同拼写？
3. **related_query（相关技术）**：是否扩展了合理的上下游技术或紧密相关的子领域？
4. **survey_query（综述覆盖）**：是否正确组合了核心词与 survey/review 等综述类限定词？

## 输出格式（严格 JSON）
{{
    "exact_query_score": 整数,
    "exact_query_reason": "不超过 20 字",
    "synonym_query_score": 整数,
    "synonym_query_reason": "不超过 20 字",
    "related_query_score": 整数,
    "related_query_reason": "不超过 20 字",
    "survey_query_score": 整数,
    "survey_query_reason": "不超过 20 字",
    "overall_assessment": "总体评价，不超过 50 字"
}}
"""

    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "你是严格的学术检索评估专家，只输出 JSON。"},
                {"role": "user", "content": eval_prompt}
            ],
            temperature=0.1
        )
        raw = response.choices[0].message.content
        llm_result = extract_json_from_text(raw)
    except Exception as e:
        print(f"  ⚠️ LLM 评估失败，使用兜底分数: {e}")
        llm_result = {}

    # 4. 计算总分
    dims = ["exact_query", "synonym_query", "related_query", "survey_query"]
    llm_total = sum(llm_result.get(f"{d}_score", 3) for d in dims)
    total_score = llm_total + avg_syntax

    return {
        "topic": topic,
        "ideal_comment": comment,
        "exact_query_score": llm_result.get("exact_query_score", 3),
        "exact_query_reason": llm_result.get("exact_query_reason", ""),
        "synonym_query_score": llm_result.get("synonym_query_score", 3),
        "synonym_query_reason": llm_result.get("synonym_query_reason", ""),
        "related_query_score": llm_result.get("related_query_score", 3),
        "related_query_reason": llm_result.get("related_query_reason", ""),
        "survey_query_score": llm_result.get("survey_query_score", 3),
        "survey_query_reason": llm_result.get("survey_query_reason", ""),
        "syntax_score": round(avg_syntax, 1),
        "syntax_reason": f"各 query 语法分: {syntax_detail}",
        "total_score": round(total_score, 1),
        "max_score": 25,
        "overall_assessment": llm_result.get("overall_assessment", ""),
        "actual_queries": actual,
    }


# ==========================================
# 打印函数
# ==========================================

def print_result(result: dict):
    """打印单个 Topic 的评估结果"""
    score = result.get('total_score', 0)
    max_s = result.get('max_score', 25)

    if score >= 20:      emoji = "✅"
    elif score >= 15:    emoji = "⚠️"
    else:                emoji = "❌"

    print(f"\n  {emoji} 评分：{score:.1f}/{max_s}")
    print(f"     💬 {result.get('overall_assessment', '')}")
    print(f"     ┌────────────────────────────────┐")
    print(f"     │ exact_query  (精确) : {result.get('exact_query_score', '?')}/5")
    print(f"     │ synonym_query (同义) : {result.get('synonym_query_score', '?')}/5")
    print(f"     │ related_query (相关) : {result.get('related_query_score', '?')}/5")
    print(f"     │ survey_query  (综述) : {result.get('survey_query_score', '?')}/5")
    print(f"     │ syntax       (语法) : {result.get('syntax_score', '?')}/5")
    print(f"     └────────────────────────────────┘")

    reasons = [
        f"     📝 exact: {result.get('exact_query_reason', '')}",
        f"     📝 syno:  {result.get('synonym_query_reason', '')}",
        f"     📝 rela:  {result.get('related_query_reason', '')}",
        f"     📝 surv:  {result.get('survey_query_reason', '')}",
        f"     📝 synt:  {result.get('syntax_reason', '')}",
    ]
    for r in reasons:
        if r.strip() and r[-1] != ':':
            print(r)


# ==========================================
# 主入口
# ==========================================

def main():
    print("=" * 55)
    print("  📊 Planner Query 改写质量评估")
    print("  评估方式：人工标注标准答案 + LLM 对比判分")
    print("  语法检查：硬编码正则（不依赖 LLM）")
    print("=" * 55)

    all_results = []

    for i, case in enumerate(TEST_CASES, 1):
        print(f"\n  [{i}/{len(TEST_CASES)}] {case['topic']}")
        result = evaluate_single_case(
            topic=case["topic"],
            ideal=case["ideal"],
            comment=case["comment"]
        )
        print_result(result)
        all_results.append(result)

    # 汇总
    print("\n" + "=" * 55)
    print("  📈 汇总报告")
    print("=" * 55)

    total_scores = [r.get("total_score", 0) for r in all_results]
    avg_score = sum(total_scores) / len(total_scores) if total_scores else 0
    max_total = all_results[0].get("max_score", 25) if all_results else 25

    print(f"\n  平均分: {avg_score:.1f}/{max_total}")
    print(f"  最高分: {max(total_scores):.1f}/{max_total}")
    print(f"  最低分: {min(total_scores):.1f}/{max_total}")

    if avg_score >= 20:
        print(f"\n  ✅ Planner Query 改写质量优秀")
    elif avg_score >= 15:
        print(f"\n  ⚠️ 质量一般，建议优化 Prompt")
    else:
        print(f"\n  ❌ 质量较差，需要重点优化")

    # 按维度汇总
    dims = ["exact_query", "synonym_query", "related_query", "survey_query", "syntax"]
    dim_labels = ["精确匹配", "同义扩展", "相关覆盖", "综述覆盖", "语法合规"]

    print(f"\n  ┌──────────────────────────────────────────────┐")
    for dim, label in zip(dims, dim_labels):
        scores = [r.get(f"{dim}_score", 0) for r in all_results]
        avg_dim = sum(scores) / len(scores) if scores else 0
        bar = "█" * max(0, min(5, int(avg_dim))) + "░" * max(0, 5 - max(0, min(5, int(avg_dim))))
        print(f"  │ {label:<8s} {avg_dim:.1f}/5  {bar} │")
    print(f"  └──────────────────────────────────────────────┘")

    # 优化建议
    print(f"\n  💡 优化建议：")
    weak = []
    for dim, label in zip(dims, dim_labels):
        scores = [r.get(f"{dim}_score", 0) for r in all_results]
        if scores:
            avg_dim = sum(scores) / len(scores)
            if avg_dim < 3.5:
                weak.append(label)
    if weak:
        print(f"     🟡 薄弱维度：{', '.join(weak)}")
        print(f"     建议：调整 Planner Prompt 中对这些维度的要求")
    else:
        print(f"     ✅ 所有维度表现良好，无需优化")

    print()


if __name__ == "__main__":
    main()