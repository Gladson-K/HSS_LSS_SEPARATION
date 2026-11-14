"""Free-tier AI agent with provider fallback.

Priority order (configurable): ollama (local, unlimited & free) -> groq ->
gemini -> openrouter -> offline. If no provider is reachable the agent returns a
deterministic, grounded answer built from the analysis context so the product
still works without any API keys.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

from app.config import get_settings
from app.core.schemas import AgentResponse

SYSTEM_PROMPT = (
    "You are CardioPulmo AI, a careful auscultation analysis assistant. "
    "You explain heart/lung sound separation results in plain language, give "
    "evidence-based lifestyle suggestions, and always recommend consulting a "
    "clinician. Never claim to diagnose. Be concise (max 150 words)."
)


def _post_json(url: str, payload: dict, headers: dict | None = None,
               timeout: float = 20.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", "application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return json.loads(resp.read().decode("utf-8"))


def _build_messages(question: str, context: dict | None) -> list[dict]:
    ctx = ""
    if context:
        ctx = "\nAnalysis context:\n" + json.dumps(context, indent=2)[:4000]
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question + ctx},
    ]


# --------------------------- providers ---------------------------

def _ask_ollama(messages: list[dict], s) -> str:
    payload = {"model": s.ollama_model, "messages": messages, "stream": False}
    out = _post_json(f"{s.ollama_url.rstrip('/')}/api/chat", payload)
    return out.get("message", {}).get("content", "").strip()


def _ask_groq(messages: list[dict], s) -> str:
    if not s.groq_api_key:
        raise RuntimeError("no groq key")
    out = _post_json(
        "https://api.groq.com/openai/v1/chat/completions",
        {"model": s.groq_model, "messages": messages, "max_tokens": 400},
        {"Authorization": f"Bearer {s.groq_api_key}"},
    )
    return out["choices"][0]["message"]["content"].strip()


def _ask_gemini(messages: list[dict], s) -> str:
    if not s.gemini_api_key:
        raise RuntimeError("no gemini key")
    user_text = "\n".join(m["content"] for m in messages)
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{s.gemini_model}:generateContent?key={s.gemini_api_key}"
    )
    out = _post_json(url, {"contents": [{"parts": [{"text": user_text}]}]})
    return out["candidates"][0]["content"]["parts"][0]["text"].strip()


def _ask_openrouter(messages: list[dict], s) -> str:
    if not s.openrouter_api_key:
        raise RuntimeError("no openrouter key")
    out = _post_json(
        "https://openrouter.ai/api/v1/chat/completions",
        {"model": s.openrouter_model, "messages": messages, "max_tokens": 400},
        {
            "Authorization": f"Bearer {s.openrouter_api_key}",
            "HTTP-Referer": "https://github.com/Gladson-K/HeartLungSound_NAS_Separation",
            "X-Title": "CardioPulmo AI",
        },
    )
    return out["choices"][0]["message"]["content"].strip()


_PROVIDERS = {
    "ollama": _ask_ollama,
    "groq": _ask_groq,
    "gemini": _ask_gemini,
    "openrouter": _ask_openrouter,
}


def _offline_answer(question: str, context: dict | None) -> str:
    ctx = context or {}
    sep = ctx.get("separation", {}) if isinstance(ctx, dict) else {}
    graphs = ctx.get("graphs", []) if isinstance(ctx, dict) else []
    flagged = [g for g in graphs if g.get("severity") in ("warning", "critical")] if graphs else []
    lines = ["Offline analysis summary (no LLM provider configured):"]
    if sep:
        lines.append(
            f"- Screen: {sep.get('dominant_diagnosis', 'n/a')} "
            f"(confidence {round(float(sep.get('diagnosis_confidence', 0))*100)}%)."
        )
        if sep.get("heartbeat_bpm"):
            lines.append(f"- Heart rate: {sep['heartbeat_bpm']:.0f} BPM.")
        if sep.get("breath_rate_bpm"):
            lines.append(f"- Respiratory rate: {sep['breath_rate_bpm']:.0f} breaths/min.")
    if flagged:
        lines.append(f"- {len(flagged)} graph(s) need attention: "
                     + ", ".join(g.get("title", "graph") for g in flagged) + ".")
    lines.append(
        "General advice: stay active, eat a low-sodium diet, avoid smoking, sleep "
        "well, and consult a clinician for any persistent symptoms."
    )
    lines.append("Question received: " + question)
    return "\n".join(lines)


def ask(question: str, context: dict | None = None) -> AgentResponse:
    settings = get_settings()
    messages = _build_messages(question, context)
    for provider in settings.agent_provider_order:
        if provider == "offline":
            continue
        fn = _PROVIDERS.get(provider)
        if fn is None:
            continue
        try:
            text = fn(messages, settings)
            if text:
                model = {
                    "ollama": settings.ollama_model,
                    "groq": settings.groq_model,
                    "gemini": settings.gemini_model,
                    "openrouter": settings.openrouter_model,
                }.get(provider, provider)
                return AgentResponse(answer=text, provider=provider, model=model,
                                     used_fallback=provider != settings.agent_provider_order[0])
        except (urllib.error.URLError, urllib.error.HTTPError, KeyError, IndexError,
                ValueError, TimeoutError, RuntimeError):
            continue
    return AgentResponse(answer=_offline_answer(question, context),
                         provider="offline", model="rule-based", used_fallback=True)


def available_providers() -> list[str]:
    settings = get_settings()
    out = ["offline"]
    if settings.groq_api_key:
        out.append("groq")
    if settings.gemini_api_key:
        out.append("gemini")
    if settings.openrouter_api_key:
        out.append("openrouter")
    # ollama is checked lazily; report as configured
    out.append("ollama")
    return out
