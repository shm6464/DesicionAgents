"""Streamlit Web UI for DecisionAgents.

A lightweight browser front-end over the existing analysis pipeline. It reuses
``TradingAgentsGraph.propagate()`` and ``save_reports()`` — it does NOT
reimplement any analysis logic — and only collects the inputs those entry
points already accept (ticker, date, analysts, models, debate/risk rounds).

Run it from the project root with:

    streamlit run webapp.py

Install the UI extras first:

    pip install "tradingagents[ui]"

The provider, models, rounds and every other knob are read from the same
``DEFAULT_CONFIG`` / ``TRADINGAGENTS_*`` environment, so a run made here uses
the exact same configuration as a programmatic ``propagate()`` call.
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta

import streamlit as st

from tradingagents.agents.rating import is_review, run_rating
from tradingagents.dataflows.date_window import get_current_date
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.llm_clients.model_catalog import (
    get_cn_dashscope_tiers,
    get_cn_default_model,
    get_model_options,
)

logger = logging.getLogger(__name__)

ANALYST_LABELS = {
    "market": "Market Analyst (行情)",
    "social": "Sentiment Analyst (舆情)",
    "news": "News Analyst (新闻)",
    "fundamentals": "Fundamentals Analyst (基本面)",
}

# Provider ids the UI offers; the id is what DEFAULT_CONFIG / llm_clients use.
PROVIDER_OPTIONS = [
    "openai", "deepseek", "qwen", "qwen-cn", "glm", "glm-cn", "kimi",
    "minimax", "minimax-cn", "anthropic", "google", "azure", "xai",
    "openrouter", "mistral", "groq", "nvidia", "ollama", "openai_compatible",
]


def _model_options(provider: str, mode: str) -> list[str]:
    """Model ids for a provider, tolerating a provider without a catalog entry."""
    try:
        return [model_id for _label, model_id in get_model_options(provider, mode)]
    except (KeyError, ValueError):
        return ["custom"]


# Text LLMs the user's DashScope (qwen-cn) account currently holds free quota
# for, split quick/deep. This is now sourced from the authoritative domestic
# tier table in ``llm_clients.model_catalog`` (stage 4) rather than hardcoded
# here, so the dropdown and the cost tracker's prices stay in lock-step.
# Filtered for this pipeline: pure-text models with tool-calling support. The
# catalog's qwen3.8-flash is exhausted, so these are the practical choices.
# Deliberately excluded: qwen-vl-*/qvq-*/ocr (vision), qwen-math-*/coder-*/mt
# (specialized), *-character (roleplay), deepseek-r1* (no function calling),
# *-thinking* (reasoning_content round-trip is not handled by the qwen client).
_DASHSCOPE_PROVIDERS = ("qwen", "qwen-cn")


def _dashscope_free_models() -> list[str]:
    """All free-quota DashScope models (quick + deep merged), from the authoritative table."""
    tiers = get_cn_dashscope_tiers()
    return list(tiers.get("quick", [])) + list(tiers.get("deep", []))


def _default_model(provider: str, mode: str) -> str:
    """Recommended DashScope model for a tier, falling back to a sensible id."""
    return get_cn_default_model(provider, mode) or "qwen-flash"


def _model_choices(provider: str) -> list[str]:
    """All catalog options for a provider, quick and deep tiers merged.

    The quick and deep pickers share the same model pool, so any model can be
    assigned to either tier (e.g. run the Trader on a deep model while the
    analysts stay on a quick one, or the reverse). Order is preserved and
    duplicates are dropped; "custom" stays so a free-text id is still possible.
    """
    choices: list[str] = []
    for mode in ("quick", "deep"):
        for model_id in _model_options(provider, mode):
            if model_id not in choices:
                choices.append(model_id)
    if provider in _DASHSCOPE_PROVIDERS:
        for model_id in _dashscope_free_models():
            if model_id not in choices:
                choices.append(model_id)
    # Keep the free-text "custom" entry last, whichever tier introduced it.
    if "custom" in choices:
        choices.remove("custom")
        choices.append("custom")
    return choices


def _pick_model(label: str, choices: list[str], recommended: str, key: str) -> str:
    """A model dropdown that falls back to free-text input for 'custom'."""
    index = choices.index(recommended) if recommended in choices else 0
    choice = st.selectbox(label, choices, index=index, key=f"{key}_select")
    if choice == "custom":
        return st.text_input(
            f"{label} — 手动输入模型 ID",
            value="" if recommended == "custom" else recommended,
            key=f"{key}_custom",
        )
    return choice


def _build_config(provider: str, deep_llm: str, quick_llm: str,
                  debate_rounds: int, risk_rounds: int, output_language: str) -> dict:
    """Assemble a run config from the form, starting from the env-aware defaults."""
    config = DEFAULT_CONFIG.copy()
    config["llm_provider"] = provider
    config["deep_think_llm"] = deep_llm
    config["quick_think_llm"] = quick_llm
    config["max_debate_rounds"] = int(debate_rounds)
    config["max_risk_discuss_rounds"] = int(risk_rounds)
    config["output_language"] = output_language
    return config


def _render_analyst_reports(final_state: dict):
    """Show each analyst report the run produced, in its own expander."""
    reports = [
        ("market_report", "Market Analyst"),
        ("sentiment_report", "Sentiment Analyst"),
        ("news_report", "News Analyst"),
        ("fundamentals_report", "Fundamentals Analyst"),
    ]
    for key, title in reports:
        content = final_state.get(key)
        if content:
            with st.expander(f"📊 {title}", expanded=False):
                st.markdown(content)


def _render_debate(final_state: dict):
    """Show the bull/bear debate and research-manager decision."""
    debate = final_state.get("investment_debate_state") or {}
    bull = debate.get("bull_history", "")
    bear = debate.get("bear_history", "")
    if bull or bear:
        with st.expander("⚔️ 多空研究员辩论", expanded=False):
            if bull:
                st.markdown("**Bull Researcher（多方）**")
                st.markdown(bull)
            if bear:
                st.markdown("**Bear Researcher（空方）**")
                st.markdown(bear)
    judge = final_state.get("investment_plan")
    if judge:
        with st.expander("🧭 Research Manager 决策", expanded=False):
            st.markdown(judge)


def _render_risk_and_final(final_state: dict):
    """Show the risk debate and the portfolio manager's final call."""
    risk = final_state.get("risk_debate_state") or {}
    for key, title in [
        ("aggressive_history", "Aggressive Analyst（激进）"),
        ("conservative_history", "Conservative Analyst（保守）"),
        ("neutral_history", "Neutral Analyst（中性）"),
    ]:
        content = risk.get(key, "")
        if content:
            with st.expander(f"🛡️ {title}", expanded=False):
                st.markdown(content)
    decision = final_state.get("final_trade_decision", "")
    if decision:
        with st.expander("🎯 最终交易决策（Portfolio Manager）", expanded=True):
            st.markdown(decision)


def _final_rating_badge(final_state: dict):
    rating = run_rating(final_state)
    if is_review(rating):
        st.warning(f"最终评级：{rating}（决策文本中未能识别出有效评级，需人工复核）")
        return
    st.success(f"最终评级：**{rating}**")


def _render_cost_summary(graph: TradingAgentsGraph):
    """Show token usage and estimated cost for the run."""
    try:
        summary = graph.cost_summary()
    except Exception as exc:
        st.caption(f"成本统计不可用：{exc}")
        return

    total_calls = summary.get("total_calls", 0)
    if not total_calls:
        st.caption("本次运行未产生可计费的 LLM 调用（可能全部命中缓存）。")
        return

    st.subheader("💸 Token 用量与成本")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("LLM 调用次数", total_calls)
    col2.metric("输入 tokens", f"{summary.get('total_input_tokens', 0):,}")
    col3.metric("输出 tokens", f"{summary.get('total_output_tokens', 0):,}")
    col4.metric("估算成本", f"${summary.get('total_cost_usd', 0):.4f}")

    by_tier = summary.get("by_tier") or {}
    if by_tier:
        st.markdown("**按层级（deep/quick）**")
        rows = []
        for tier, data in by_tier.items():
            rows.append({
                "层级": tier,
                "调用": data.get("calls", 0),
                "输入 tokens": f"{data.get('input_tokens', 0):,}",
                "输出 tokens": f"{data.get('output_tokens', 0):,}",
                "成本 (USD)": f"${data.get('cost', 0):.4f}",
            })
        st.dataframe(rows, use_container_width=True, hide_index=True)

    by_model = summary.get("by_model") or {}
    if by_model:
        st.markdown("**按模型**")
        model_rows = []
        for model, data in by_model.items():
            model_rows.append({
                "模型": model,
                "调用": data.get("calls", 0),
                "输入 tokens": f"{data.get('input_tokens', 0):,}",
                "输出 tokens": f"{data.get('output_tokens', 0):,}",
                "成本 (USD)": f"${data.get('cost', 0):.4f}",
            })
        st.dataframe(model_rows, use_container_width=True, hide_index=True)


def _save_and_report(final_state: dict, graph: TradingAgentsGraph, ticker: str):
    """Save the report tree under results_dir and show the path."""
    try:
        save_path = graph.save_reports(final_state, ticker)
        st.info(f"报告已保存到：`{save_path}`")
    except Exception as exc:  # saving must not hide an otherwise-good run
        st.warning(f"报告保存失败：{exc}")


def main():
    st.set_page_config(page_title="DecisionAgents 多智能体决策分析", layout="wide")
    st.title("📈 DecisionAgents 多智能体决策系统")
    st.caption(
        "DecisionAgents：基于 LangGraph 的多智能体决策流水线。"
        "复用 `TradingAgentsGraph.propagate()` / `save_reports()` 完成端到端分析。"
    )

    with st.sidebar:
        st.header("分析参数")

        ticker = st.text_input(
            "股票代码 (ticker)",
            value="NVDA",
            help="如 NVDA、0700.HK、600519.SS。会走 yfinance 的 symbol 归一化。",
        )

        today = get_current_date()
        default_date = (datetime.strptime(today, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        trade_date = st.text_input(
            "分析日期 (YYYY-MM-DD)",
            value=default_date,
            help="不能晚于今天；回测/历史日期会触发 point-in-time 数据过滤。",
        )

        st.subheader("分析师团队")
        selected_analysts = []
        for key, label in ANALYST_LABELS.items():
            if st.checkbox(label, value=True, key=f"analyst_{key}"):
                selected_analysts.append(key)
        if not selected_analysts:
            st.warning("请至少选择一个分析师")

        st.subheader("模型与参数")
        provider = st.selectbox("LLM Provider", PROVIDER_OPTIONS,
                                index=PROVIDER_OPTIONS.index("qwen-cn"))
        # Quick and deep pickers share the same model pool, so any model can be
        # assigned to either tier (e.g. run the Trader on a deep model while the
        # analysts stay on a quick one). Only the recommended default differs.
        models = _model_choices(provider)
        deep_llm = _pick_model("Deep-think 模型（强推理）", models,
                               _default_model(provider, "deep"), "deep")
        quick_llm = _pick_model("Quick-think 模型（快速便宜）", models,
                                _default_model(provider, "quick"), "quick")

        debate_rounds = st.number_input(
            "辩论轮数 (max_debate_rounds)", min_value=1, max_value=10,
            value=int(DEFAULT_CONFIG["max_debate_rounds"]),
        )
        risk_rounds = st.number_input(
            "风控讨论轮数 (max_risk_discuss_rounds)", min_value=1, max_value=10,
            value=int(DEFAULT_CONFIG["max_risk_discuss_rounds"]),
        )
        # Default to English to match DEFAULT_CONFIG["output_language"], so runs
        # through the Web UI and programmatic calls produce the same language. The old default (index=1 -> "Chinese") let the research /
        # portfolio managers emit Chinese while the analysts (served from the
        # English response cache) stayed English, producing a mixed report.
        output_language = st.selectbox("报告语言", ["English", "Chinese"], index=0)

        run_clicked = st.button("🚀 开始分析", type="primary", use_container_width=True)

    st.header("分析结果")
    if not run_clicked:
        st.info("在左侧配置参数后点击「开始分析」。首次运行会调用 LLM 与行情接口，耗时较长。")
        return

    if not selected_analysts:
        st.error("未选择任何分析师，无法开始。")
        return

    config = _build_config(provider, deep_llm, quick_llm, debate_rounds, risk_rounds, output_language)

    # Show what is about to run, mirroring the pre-run summary.
    st.markdown(
        f"- **Ticker**: `{ticker}` · **日期**: `{trade_date}`\n"
        f"- **Provider**: `{provider}` · deep=`{deep_llm}` · quick=`{quick_llm}`\n"
        f"- **分析师**: {', '.join(selected_analysts)} · 辩论 {debate_rounds} 轮 · 风控 {risk_rounds} 轮"
    )

    progress = st.progress(0.0, text="初始化 graph…")
    start_time = time.time()

    try:
        graph = TradingAgentsGraph(tuple(selected_analysts), config=config, debug=False)
        progress.progress(0.15, text="正在运行多智能体流水线…")

        final_state, signal = graph.propagate(ticker, trade_date)

        progress.progress(1.0, text="分析完成")
        elapsed = time.time() - start_time
        st.caption(f"耗时 {elapsed:.1f}s")

        _final_rating_badge(final_state)
        _render_cost_summary(graph)
        _render_analyst_reports(final_state)
        _render_debate(final_state)
        _render_risk_and_final(final_state)
        _save_and_report(final_state, graph, ticker)

    except Exception as exc:
        progress.empty()
        logger.exception("analysis failed")
        st.error(f"分析失败：{exc}")


if __name__ == "__main__":
    main()
