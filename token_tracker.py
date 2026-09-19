"""
Token 使用统计模块
功能：追踪所有 Agent 的 LLM 调用 token 消耗，汇总输出费用估算
用法：每个 Agent 在 LLM 调用后调用 tracker.record() 记录 token 用量
"""

import time
from typing import Optional


class TokenRecord:
    """单次 LLM 调用的 token 记录"""
    def __init__(self, agent: str, model: str, prompt_tokens: int, 
                 completion_tokens: int, operation: str = ""):
        self.agent = agent
        self.model = model
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.operation = operation
        self.timestamp = time.strftime("%H:%M:%S")
    
    def __str__(self):
        op = f" [{self.operation}]" if self.operation else ""
        return (f"  {self.agent}{op}: "
                f"Prompt={self.prompt_tokens:>6}  "
                f"Completion={self.completion_tokens:>6}  "
                f"Total={self.total_tokens:>6}  "
                f"Model={self.model}")


class TokenTracker:
    """全局 Token 跟踪器，收集所有 Agent 的 token 消耗"""
    
    # DeepSeek V4 Flash 参考价格（元/百万 token）
    PRICING = {
        "deepseek-chat": {"prompt": 0.5, "completion": 1.5},
        "deepseek-v4-flash": {"prompt": 0.5, "completion": 1.5},
        "DEFAULT": {"prompt": 0.5, "completion": 1.5},
    }
    
    def __init__(self):
        self.records: list[TokenRecord] = []
    
    def record(self, agent: str, model: str, prompt_tokens: int, 
               completion_tokens: int, operation: str = ""):
        """记录一次 LLM 调用的 token 用量"""
        record = TokenRecord(agent, model, prompt_tokens, completion_tokens, operation)
        self.records.append(record)
        return record
    
    @property
    def total_prompt_tokens(self) -> int:
        return sum(r.prompt_tokens for r in self.records)
    
    @property
    def total_completion_tokens(self) -> int:
        return sum(r.completion_tokens for r in self.records)
    
    @property
    def total_tokens(self) -> int:
        return self.total_prompt_tokens + self.total_completion_tokens
    
    @property
    def total_calls(self) -> int:
        return len(self.records)
    
    def estimate_cost(self) -> dict:
        """估算总费用"""
        total_cost = 0.0
        cost_detail = {}
        for r in self.records:
            pricing = self.PRICING.get(r.model, self.PRICING["DEFAULT"])
            cost = (r.prompt_tokens / 1_000_000 * pricing["prompt"] +
                    r.completion_tokens / 1_000_000 * pricing["completion"])
            total_cost += cost
            if r.agent not in cost_detail:
                cost_detail[r.agent] = 0.0
            cost_detail[r.agent] += cost
        return {"total": total_cost, "detail": cost_detail}
    
    def summary(self) -> dict:
        """生成完整统计摘要"""
        cost = self.estimate_cost()
        return {
            "total_calls": self.total_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost": cost["total"],
            "cost_by_agent": cost["detail"],
        }
    
    def print_summary(self):
        """打印完整统计报告"""
        summary = self.summary()
        print("\n" + "=" * 70)
        print("  📊 Token 使用统计报告")
        print("=" * 70)
        
        # 按 Agent 分组显示
        by_agent = {}
        for r in self.records:
            by_agent.setdefault(r.agent, []).append(r)
        
        for agent, records in by_agent.items():
            agent_prompt = sum(r.prompt_tokens for r in records)
            agent_completion = sum(r.completion_tokens for r in records)
            print(f"\n  🤖 {agent} (共 {len(records)} 次调用)")
            for r in records:
                print(f"    {r}")
            print(f"    小计: Prompt={agent_prompt} + Completion={agent_completion} = {agent_prompt + agent_completion}")
        
        print("-" * 70)
        print(f"  总调用次数: {summary['total_calls']}")
        print(f"  总 Prompt Tokens: {summary['total_prompt_tokens']:,}")
        print(f"  总 Completion Tokens: {summary['total_completion_tokens']:,}")
        print(f"  总 Tokens: {summary['total_tokens']:,}")
        print(f"  预估费用: ¥{summary['estimated_cost']:.4f} 元")
        print(f"  (按 DeepSeek V4 Flash 价格估算)")
        print("=" * 70 + "\n")


# 全局单例，所有 Agent 共享
tracker = TokenTracker()