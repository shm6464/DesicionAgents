"""Token usage and cost accounting for LLM calls across the whole pipeline.

Stage 3 of the engineering-hardening roadmap: the framework previously had no
visibility into how many tokens each run consumed or what it cost, so a user
could not tell which model / tier / node was driving spend. This module adds:

- ``CostTracker`` — a per-run ledger of every LLM call (provider, model, tier,
  input/output tokens, latency, estimated cost).
- ``CostCallbackHandler`` — a LangChain callback that records calls made by
  providers that do NOT route through ``NormalizedChatOpenAI`` (Anthropic,
  Google, Azure, Bedrock). The OpenAI-compatible family (OpenAI, DeepSeek, Qwen,
  GLM, MiniMax, ...) is instead recorded directly inside
  ``NormalizedChatOpenAI.invoke`` — the same Runnable-safe hook the response
  cache uses — because that path already has the normalized response in hand.
- A per-model price table so token counts turn into a money figure.

Cost is *estimated*: token usage comes from the provider's own
``response_metadata`` when available (OpenAI-compatible providers all report
``token_usage``), and the per-token prices are the published list prices
(USD per 1M tokens). DashScope (qwen-cn) reports ``input_tokens`` /
``output_tokens`` in the same place, so the same table covers both.

Design notes
------------
- Recording is best-effort and never breaks a run: a provider that omits
  ``token_usage`` simply contributes zero cost rather than raising.
- The tracker is tier-aware ("deep" vs "quick") so a report can show how much
  the expensive reasoning model cost versus the cheap one — the practical
  question the roadmap asks.
- A cache HIT is free: ``invoke`` short-circuits before the network call and
  never records a billing entry, so re-running a cached ticker shows the true
  marginal cost (zero for cached calls).
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger(__name__)

# Published list prices, USD per 1M tokens. ``input``/``output``. Missing models
# fall back to a zero price (their calls still count tokens, just no money).
# DashScope free-tier models (qwen-flash/turbo/plus) still carry a nominal list
# price so the report shows the *list* cost a paying account would incur; when
# running on free quota the user can zero these via ``cost_price_table``.
#
# Domestic models (Qwen / DeepSeek / GLM / Kimi) are sourced from the single
# authoritative tier+price table in ``llm_clients.model_catalog`` (stage 4), so
# the tiering and prices cannot drift between the UI, the tracker, and the CLI.
# Native OpenAI models have no domestic tier entry and stay inline here.
_OPENAI_PRICES: dict[str, dict[str, float]] = {
    "gpt-6-luna":     {"input": 0.40, "output": 1.60},
    "gpt-6-sol":      {"input": 2.00, "output": 8.00},
    "gpt-6-astra":    {"input": 2.00, "output": 8.00},
    "gpt-5.6-luna":   {"input": 0.40, "output": 1.60},
    "gpt-5.6-terra":  {"input": 1.00, "output": 4.00},
    "gpt-5.6":        {"input": 2.00, "output": 8.00},
    "gpt-5.5":        {"input": 1.25, "output": 10.00},
    "gpt-4.1":        {"input": 2.00, "output": 8.00},
    "gpt-4o":         {"input": 2.50, "output": 10.00},
    "gpt-4o-mini":    {"input": 0.15, "output": 0.60},
}


def _build_default_prices() -> dict[str, dict[str, float]]:
    """Merge the domestic tier+price table (model_catalog) with OpenAI prices.

    Lazy so importing cost_tracker never pulls the (heavier) model_catalog
    module unless a price is actually needed; model_catalog is otherwise a pure
    data module, but the split keeps the two concerns decoupled.
    """
    from tradingagents.llm_clients.model_catalog import get_cn_model_tiers

    prices: dict[str, dict[str, float]] = dict(_OPENAI_PRICES)
    for model, (_tier, price) in get_cn_model_tiers().items():
        prices[model] = price
    return prices


_DEFAULT_PRICES: dict[str, dict[str, float]] = _build_default_prices()


class CostTracker:
    """A per-run ledger of LLM call costs.

    Thread-safe enough for the graph's use: the graph runs the analysts in
    parallel subgraphs but each LLM call is recorded atomically via ``record``;
    Python's GIL makes list append safe. Reports are generated once, at the end
    of a run, after all calls have been recorded.
    """

    def __init__(self, config: dict | None = None):
        self.config = config or {}
        self.enabled = self._coerce_bool(self.config.get("cost_tracking_enabled", True))
        self.price_table = self.config.get("cost_price_table") or _DEFAULT_PRICES
        # A user-supplied price table overrides defaults on a per-model basis;
        # merge so a partial override doesn't drop the built-in rows.
        self.price_table = {**_DEFAULT_PRICES, **self.price_table}
        self.calls: list[dict[str, Any]] = []
        self.started_at = time.time()

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() not in ("false", "0", "no", "off", "")
        return bool(value)

    def _price(self, model: str) -> dict[str, float]:
        return self.price_table.get(model, {"input": 0.0, "output": 0.0})

    @staticmethod
    def _tokens(usage: Any) -> tuple[int, int]:
        """Pull ``(input_tokens, output_tokens)`` from a token_usage mapping.

        OpenAI-compatible providers (and DashScope) put the same keys in
        ``response_metadata["token_usage"]``; handle the common aliases so a
        provider that names them slightly differently still counts.
        """
        if not usage:
            return 0, 0
        if isinstance(usage, dict):
            get = usage.get
        else:
            get = lambda k: getattr(usage, k, None)  # noqa: E731
        input_tokens = get("input_tokens") or get("prompt_tokens") or 0
        output_tokens = get("output_tokens") or get("completion_tokens") or 0
        return int(input_tokens or 0), int(output_tokens or 0)

    def record(self, *, provider: str, model: str, tier: str,
               input_tokens: int = 0, output_tokens: int = 0,
               latency: float = 0.0) -> None:
        """Record one LLM call. ``tier`` is ``"deep"`` or ``"quick"``."""
        if not self.enabled:
            return
        price = self._price(model)
        input_cost = input_tokens / 1_000_000 * price["input"]
        output_cost = output_tokens / 1_000_000 * price["output"]
        self.calls.append({
            "provider": provider,
            "model": model,
            "tier": tier,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "latency": round(latency, 3),
            "input_cost": round(input_cost, 6),
            "output_cost": round(output_cost, 6),
            "cost": round(input_cost + output_cost, 6),
        })

    def summarize(self) -> dict[str, Any]:
        """Aggregate the ledger into a report dict (tokens, cost, latency)."""
        calls = self.calls
        total_input = sum(c["input_tokens"] for c in calls)
        total_output = sum(c["output_tokens"] for c in calls)
        total_cost = sum(c["cost"] for c in calls)
        total_latency = sum(c["latency"] for c in calls)

        by_tier: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        by_model: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost": 0.0}
        )
        for c in calls:
            for bucket, key in ((by_tier, c["tier"]), (by_model, c["model"])):
                row = bucket[key]
                row["calls"] += 1
                row["input_tokens"] += c["input_tokens"]
                row["output_tokens"] += c["output_tokens"]
                row["cost"] += c["cost"]

        return {
            "elapsed_seconds": round(time.time() - self.started_at, 3),
            "total_calls": len(calls),
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_tokens": total_input + total_output,
            "total_cost_usd": round(total_cost, 6),
            "total_latency_seconds": round(total_latency, 3),
            "by_tier": {k: dict(v) for k, v in sorted(by_tier.items())},
            "by_model": {k: dict(v) for k, v in sorted(by_model.items())},
        }

    def write_report(self, report_dir: str | os.PathLike) -> Path:
        """Write ``cost_report.json`` under ``report_dir`` and return its path."""
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)
        path = report_dir / "cost_report.json"
        path.write_text(
            json.dumps(self.summarize(), indent=4, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("cost report written to %s", path)
        return path


class CostCallbackHandler(BaseCallbackHandler):
    """LangChain callback that records calls for non-OpenAI providers.

    Anthropic / Google / Azure / Bedrock use their own client classes (not
    ``NormalizedChatOpenAI``), so they are not caught by the invoke-level hook.
    Passing this handler through ``build_llm_kwargs``'s ``callbacks`` channel
    records their usage via LangChain's ``on_llm_end``, which carries a
    ``token_usage`` in the response's LLM output / metadata.
    """

    def __init__(self, tracker: CostTracker, provider: str, tier: str):
        self.tracker = tracker
        self.provider = provider
        self.tier = tier

    def on_llm_end(self, response, **kwargs: Any) -> None:
        usage = None
        # LangChain surfaces token usage in several shapes depending on the
        # provider; walk the generations and metadata to find one.
        try:
            for gen in response.generations:
                for chunk in gen:
                    meta = getattr(chunk, "generation_info", None) or {}
                    usage = usage or meta.get("usage") or meta.get("token_usage")
            if usage is None:
                llm_output = getattr(response, "llm_output", None) or {}
                usage = llm_output.get("token_usage")
        except Exception:
            usage = None
        input_tokens, output_tokens = self.tracker._tokens(usage)
        model = getattr(getattr(response, "llm_output", None), "get", lambda *_: "")(  # noqa: B009
            "model_name", ""
        ) or ""
        self.tracker.record(
            provider=self.provider,
            model=model or "unknown",
            tier=self.tier,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )


# Process-wide singleton, mirrored from cache.py so the invoke-level hook in
# openai_client.py (which runs inside graph tool calls and holds no graph ref)
# reaches the same tracker the graph built.
_module_tracker: CostTracker | None = None

# Model id -> tier ("deep"/"quick") mapping, registered by the graph at startup
# so the invoke-level hook labels each call with the *authoritative* tier rather
# than a name heuristic. Shared across the process because the hook is detached
# from the graph object.
_model_tier_map: dict[str, str] = {}


def set_model_tier(model: str, tier: str) -> None:
    """Register the tier for a model id (the graph calls this for deep/quick)."""
    _model_tier_map[model] = tier


def get_model_tier(model: str) -> str | None:
    """The registered tier for a model id, or None if unregistered."""
    return _model_tier_map.get(model)


def get_cost_tracker(config: dict | None = None) -> CostTracker:
    """The process-wide cost tracker, built lazily from ``config``."""
    global _module_tracker
    if _module_tracker is None:
        _module_tracker = CostTracker(config)
    return _module_tracker


def set_cost_tracker(tracker: CostTracker) -> None:
    """Register a tracker as the process-wide one (the graph calls this)."""
    global _module_tracker
    _module_tracker = tracker


def reset_cost_tracker() -> None:
    """Drop the module-level tracker and tier map (tests/config reloads)."""
    global _module_tracker, _model_tier_map
    _module_tracker = None
    _model_tier_map = {}
