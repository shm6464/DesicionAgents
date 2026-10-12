"""单次分析演示：用 DashScope (qwen-plus + qwen-flash) 分析 AAPL。"""
from dotenv import load_dotenv
load_dotenv()

from tradingagents import TradingAgentsGraph
from tradingagents.default_config import build_default_config

config = build_default_config()
config["llm_provider"] = "qwen-cn"
config["deep_think_llm"] = "qwen-plus"
config["quick_think_llm"] = "qwen-flash"
config["max_debate_rounds"] = 1
config["max_risk_discuss_rounds"] = 1
config["max_tool_rounds"] = 8
config["news_article_limit"] = 10
config["output_language"] = "Chinese"

print("=== 启动 Agent 分析 ===")
print("标的: AAPL (苹果)")
print("分析日期: 2025-10-10")
print("模型: qwen-plus(deep) + qwen-flash(quick)")
print()

graph = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config,
)

state, signal = graph.propagate("AAPL", "2025-10-10", asset_type="stock")

print()
print("=== 最终决策 ===")
if signal:
    for k, v in signal.items():
        print(f"  {k}: {v}")

summary = graph.cost_tracker.summarize()
print()
print("=== Token 消耗 ===")
print(f"  总调用次数: {summary['total_calls']}")
print(f"  输入token:  {summary['total_input_tokens']:,}")
print(f"  输出token:  {summary['total_output_tokens']:,}")
print(f"  总token:    {summary['total_tokens']:,}")
print(f"  总费用(USD): {summary['total_cost_usd']:.4f}")
