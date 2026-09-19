"""
Planner Agent - 检索策略规划模块
职责：接收用户的模糊主题，分析意图，生成适用于 arXiv 的结构化检索策略。
"""

import json
import re
from openai import OpenAI
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from token_tracker import tracker 
from logger import setup_logger
# ==========================================
# 1. 配置区域 (从 .env 文件自动加载，彻底消除硬编码)
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
def generate_search_strategy(user_topic: str) -> dict:
    """
    调用 LLM 分析用户主题，生成多角度检索策略
    从精确、同义、相关、综述四个角度生成检索式，覆盖不同维度的相关文献
    """
    logger = setup_logger("Planner")
    logger.info(f"\n [Planner] 正在分析主题: '{user_topic}' ...")
    
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    prompt = f"""
你是一位资深的学术文献检索专家。用户提供了一个模糊的研究主题。
你的任务是从 **4 个不同角度** 生成 arXiv 检索式，确保查全率。

用户主题："{user_topic}"

【关键要求：中文主题翻译】
如果用户主题是中文，先将其翻译成英文。对于中文中的修饰词（如"的"、"发展"、"演变"、"应用"等），
不要单独割裂成一个独立条件用 AND 连接，而要用 OR 扩展成同义词组作为限定。

示例：
- 用户输入："moe的发展"
- 正确：all:"Mixture of Experts" AND all:(development OR evolution OR advancement OR progress OR survey)
- 错误：all:"Mixture of Experts" AND all:"development"
- 用户输入："transformer 推理加速"
- 正确：all:transformer AND all:(inference acceleration OR inference speedup OR efficient inference)
- 错误：all:transformer AND all:"inference acceleration"

对于 exact_query，优先使用 ti:（标题）前缀提高精度；
对于 synonym_query 和 related_query，使用 all:（全字段）保证召回。

请输出严格的 JSON 格式，包含以下字段：

1. "exact_query": 精确检索式。用 ti: 限定标题搜索，核心概念用 OR 扩展同义词。
2. "synonym_query": 同义/近义扩展检索式。用 all: 搜索，考虑缩写、同义词、不同写法。例如 MoE ↔ Mixture of Experts。
3. "related_query": 相关技术方向检索式。用 all: 搜索，主题涉及到的上下游技术、邻近领域。
4. "survey_query": 综述/背景检索式。用 all: 搜索，专门找 survey、review、comprehensive 这类论文。
5. "exclude_terms": 排除词列表（数组格式）。用于过滤掉不相关的方向。
6. "search_rationale": 50 字以内说明检索逻辑。

【极其重要】
- 每个检索式必须为每个搜索词指定字段前缀（all:, ti: 等），否则 arXiv 会报 500 错误！
- 正确示例：ti:"Mixture of Experts" AND ti:(development OR evolution)
- 错误示例："Mixture of Experts" AND "development"

【绝对规则】：只输出合法的 JSON 对象，不要 markdown 代码块标记，不要额外解释。
"""   

    try:
        response = client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": "你是一个严格的学术检索规划助手，只输出合法的 JSON 格式。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )
        
        raw_content = response.choices[0].message.content
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0
        tracker.record(
            agent="Planner",
            model=settings.model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            operation="生成多角度检索策略"
        )
        
        # 清洗可能的 markdown 标记
        cleaned_content = re.sub(r'^```json\s*', '', raw_content.strip())
        cleaned_content = re.sub(r'\s*```$', '', cleaned_content.strip())
        
        # 尝试解析 JSON
        strategy = json.loads(cleaned_content)
        
        # 确保所有必需字段存在
        required = ["exact_query", "synonym_query", "related_query", "survey_query"]
        for field in required:
            if field not in strategy:
                strategy[field] = f'all:({user_topic})'
        
        # 添加元数据
        strategy["original_topic"] = user_topic
        strategy["suggested_max_papers"] = 15
        
        # 格式化输出
        print(f"\n  📋 多角度检索策略：")
        print(f"     🎯 精确: {strategy.get('exact_query', '')[:70]}...")
        print(f"     🔄 同义: {strategy.get('synonym_query', '')[:70]}...")
        print(f"     🔗 相关: {strategy.get('related_query', '')[:70]}...")
        print(f"     📖 综述: {strategy.get('survey_query', '')[:70]}...")
        if strategy.get("exclude_terms"):
            print(f"     🚫 排除: {', '.join(strategy['exclude_terms'][:3])}")
        print(f"     💡 策略说明: {strategy.get('search_rationale', '')}")
        print()
        
        return strategy

    except (json.JSONDecodeError, Exception) as e:
        logger.error(f"检索策略生成失败: {e}")
        # 出错了就用兜底策略
        fallback = {
            "exact_query": f'all:({user_topic})',
            "synonym_query": f'all:({user_topic})',
            "related_query": f'all:({user_topic})',
            "survey_query": f'all:({user_topic}) AND all:(survey OR review)',
            "exclude_terms": [],
            "search_rationale": f"LLM 解析失败，直接使用原始主题: {user_topic}",
            "original_topic": user_topic,
            "suggested_max_papers": 15
        }
        return fallback



# ==========================================
# 3. 独立测试入口
# ==========================================
if __name__ == "__main__":
    test_topic = input("请输入测试主题: ")
    logger = setup_logger("Planner")
    result = generate_search_strategy(test_topic)
    logger.info("\n生成的策略:")
    logger.info(json.dumps(result, indent=4, ensure_ascii=False))
