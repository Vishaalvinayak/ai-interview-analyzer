"use client";

import { useState } from "react";

const SAMPLE_QUESTIONS = [
  "Explain REST API and its principles",
  "What is the difference between SQL and NoSQL databases?",
  "Explain Object-Oriented Programming concepts",
  "Describe a challenging project you worked on",
  "How do you handle tight deadlines?",
  "Explain Git version control and its importance",
  "Explain the MVC architecture pattern",
  "How would you optimize a slow database query?",
];

interface AnalysisResult {
  classification: string;
  score: number;
  confidence: { weak: number; average: number; strong: number };
  keyword_analysis: { found: string[]; missing: string[]; coverage: number };
  suggestions: string[];
  metadata: { word_count: number; has_examples: boolean };
}

export default function Home() {
  const [question, setQuestion] = useState(SAMPLE_QUESTIONS[0]);
  const [answer, setAnswer] = useState("");
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleAnalyze = async () => {
    if (answer.length < 20) {
      setError("Response too short (min 20 chars)");
      return;
    }
    setLoading(true);
    setError("");
    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, answer }),
      });
      if (!response.ok) throw new Error("Analysis failed");
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError("API Connection Error. Ensure backend is on port 8000.");
    } finally {
      setLoading(false);
    }
  };

  const getStatusColor = (type: string) => {
    if (type === "strong")
      return "text-emerald-400 border-emerald-500/30 bg-emerald-500/5";
    if (type === "average")
      return "text-amber-400 border-amber-500/30 bg-amber-500/5";
    return "text-rose-400 border-rose-500/30 bg-rose-500/5";
  };

  return (
    <div className="min-h-screen bg-[#0a0a0a] text-zinc-300 selection:bg-indigo-500/30">
      <div className="max-w-4xl mx-auto px-6 py-16">
        {/* Header */}
        <header className="mb-16 text-center">
          <h1 className="text-4xl font-light tracking-tight text-white mb-2">
            Interview{" "}
            <span className="font-semibold text-indigo-500">Intelligence</span>
          </h1>
          <p className="text-zinc-500 font-mono text-sm uppercase tracking-widest">
            DistilBERT v2.0 • Zero Latency Analysis
          </p>
        </header>

        {/* Input Area */}
        <div className="space-y-8 mb-12">
          <div className="group">
            <label className="block text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
              Selected Prompt
            </label>
            <select
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="w-full bg-zinc-900/50 border border-zinc-800 rounded-lg p-4 text-white focus:outline-none focus:border-indigo-500 transition-colors appearance-none cursor-pointer"
            >
              {SAMPLE_QUESTIONS.map((q, i) => (
                <option key={i} value={q}>
                  {q}
                </option>
              ))}
            </select>
          </div>

          <div className="relative">
            <label className="flex justify-between text-xs font-medium text-zinc-500 uppercase tracking-wider mb-3">
              <span>Your Response</span>
              <span className={answer.length > 1500 ? "text-rose-500" : ""}>
                {answer.length}/2000
              </span>
            </label>
            <textarea
              value={answer}
              onChange={(e) => setAnswer(e.target.value)}
              placeholder="Begin typing your response..."
              className="w-full bg-zinc-900/50 border border-zinc-800 rounded-xl p-6 h-64 text-lg font-light leading-relaxed focus:outline-none focus:border-indigo-500 transition-all resize-none placeholder:text-zinc-700"
            />
          </div>

          <button
            onClick={handleAnalyze}
            disabled={loading || answer.length < 20}
            className="w-full py-4 rounded-lg bg-white text-black font-semibold hover:bg-zinc-200 disabled:bg-zinc-800 disabled:text-zinc-600 transition-all flex items-center justify-center gap-2"
          >
            {loading ? (
              <div className="h-5 w-5 border-2 border-black/20 border-t-black rounded-full animate-spin" />
            ) : (
              "Run Diagnostics"
            )}
          </button>

          {error && (
            <p className="text-rose-500 text-sm text-center font-mono">
              !! {error}
            </p>
          )}
        </div>

        {/* Results View */}
        {result && (
          <div className="animate-in fade-in slide-in-from-bottom-4 duration-700 space-y-12 pt-12 border-t border-zinc-900">
            {/* Score & Verdict */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              <div className="bg-zinc-900/40 border border-zinc-800 p-8 rounded-2xl flex flex-col items-center justify-center">
                <span className="text-zinc-500 text-xs uppercase mb-2">
                  Performance Score
                </span>
                <span className="text-6xl font-bold text-white">
                  {result.score}
                </span>
              </div>

              <div
                className={`md:col-span-2 border p-8 rounded-2xl flex flex-col justify-center ${getStatusColor(result.classification)}`}
              >
                <span className="opacity-60 text-xs uppercase mb-1 font-bold">
                  AI Verdict
                </span>
                <span className="text-4xl font-light tracking-tight uppercase">
                  {result.classification} Response
                </span>
              </div>
            </div>

            {/* Suggestions & Keywords */}
            <div className="grid md:grid-cols-2 gap-12">
              <section>
                <h3 className="text-white font-medium mb-6 flex items-center gap-2">
                  <span className="h-1.5 w-1.5 bg-indigo-500 rounded-full" />
                  Key Improvements
                </h3>
                <ul className="space-y-4">
                  {result.suggestions.map((s, i) => (
                    <li
                      key={i}
                      className="text-sm text-zinc-400 leading-relaxed border-l border-zinc-800 pl-4"
                    >
                      {s}
                    </li>
                  ))}
                </ul>
              </section>

              <section>
                <h3 className="text-white font-medium mb-6 flex items-center gap-2">
                  <span className="h-1.5 w-1.5 bg-emerald-500 rounded-full" />
                  Vocabulary Coverage
                </h3>
                <div className="flex flex-wrap gap-2">
                  {result.keyword_analysis.found.map((kw, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 bg-emerald-500/10 text-emerald-400 text-xs rounded-md border border-emerald-500/20"
                    >
                      {kw}
                    </span>
                  ))}
                  {result.keyword_analysis.missing.map((kw, i) => (
                    <span
                      key={i}
                      className="px-3 py-1 bg-zinc-900 text-zinc-600 text-xs rounded-md border border-zinc-800"
                    >
                      {kw}
                    </span>
                  ))}
                </div>
              </section>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
