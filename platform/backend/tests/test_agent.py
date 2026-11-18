from app.core import ai_agent


def test_offline_agent_returns_grounded_answer():
    ctx = {
        "separation": {
            "dominant_diagnosis": "Normal",
            "diagnosis_confidence": 0.8,
            "heartbeat_bpm": 72,
            "breath_rate_bpm": 16,
        },
        "graphs": [
            {"title": "Heart Sound (HSS) Envelope", "severity": "normal"},
        ],
    }
    resp = ai_agent.ask("What should I do?", ctx)
    assert resp.answer
    assert resp.provider in ("offline", "ollama", "groq", "gemini", "openrouter")
    assert isinstance(resp.used_fallback, bool)


def test_available_providers_includes_offline():
    from app.core import ai_agent as agent

    providers = agent.available_providers()
    assert "offline" in providers
