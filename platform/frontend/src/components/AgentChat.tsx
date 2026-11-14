import { useState } from "react";
import { Bot, Loader2, Send, Sparkles } from "lucide-react";
import { api } from "@/lib/api";
import { Button } from "./ui/Button";
import { Card, CardContent, CardHeader, CardTitle } from "./ui/Card";
import { Badge } from "./ui/Badge";

interface Message {
  role: "user" | "assistant";
  text: string;
  meta?: string;
}

const QUICK = [
  "Explain my heart rate result",
  "What lifestyle changes help my lungs?",
  "When should I see a doctor?",
];

export function AgentChat({ analysisId }: { analysisId?: string }) {
  const [messages, setMessages] = useState<Message[]>([
    {
      role: "assistant",
      text:
        "Hi, I'm CardioPulmo AI. Ask me about your analysis, symptoms, or lifestyle. I use a free-tier/local model and always recommend a clinician for decisions.",
      meta: "agent",
    },
  ]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);

  async function send(question: string) {
    if (!question.trim() || busy) return;
    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setBusy(true);
    try {
      const res = await api.agent(question, analysisId);
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          text: res.answer,
          meta: `${res.provider} · ${res.model}${res.used_fallback ? " (fallback)" : ""}`,
        },
      ]);
    } catch (e) {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: e instanceof Error ? e.message : "Agent error" },
      ]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Card className="flex h-full flex-col">
      <CardHeader className="flex flex-row items-center justify-between">
        <div className="flex items-center gap-2">
          <div className="grid h-9 w-9 place-items-center rounded-xl bg-emerald-500/15 text-emerald-300">
            <Bot className="h-4 w-4" />
          </div>
          <CardTitle>AI Assistant</CardTitle>
        </div>
        <Badge>free tier · local-first</Badge>
      </CardHeader>
      <CardContent className="flex flex-1 flex-col gap-3">
        <div className="scroll-thin flex-1 space-y-3 overflow-y-auto pr-1" style={{ maxHeight: 360 }}>
          {messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-brand-600 text-white"
                    : "border border-white/10 bg-white/[0.05] text-slate-200"
                }`}
              >
                {m.text}
                {m.meta && (
                  <div className="mt-1 text-[10px] uppercase tracking-wide text-slate-400">
                    {m.meta}
                  </div>
                )}
              </div>
            </div>
          ))}
          {busy && (
            <div className="flex items-center gap-2 text-xs text-slate-400">
              <Loader2 className="h-3.5 w-3.5 animate-spin" /> thinking…
            </div>
          )}
        </div>

        <div className="flex flex-wrap gap-1.5">
          {QUICK.map((q) => (
            <button
              key={q}
              onClick={() => send(q)}
              className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-[11px] text-slate-300 hover:bg-white/10"
            >
              <Sparkles className="mr-1 inline h-3 w-3" />
              {q}
            </button>
          ))}
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="flex items-center gap-2"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your results…"
            className="h-10 flex-1 rounded-xl border border-white/10 bg-slate-950/50 px-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-brand-400 focus:outline-none"
          />
          <Button type="submit" size="icon" disabled={busy} aria-label="Send">
            <Send className="h-4 w-4" />
          </Button>
        </form>
      </CardContent>
    </Card>
  );
}
