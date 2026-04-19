"use client";

import React, { useState } from "react";

type Segment = {
  start: number;
  end: number;
  speaker: string;
  text: string;
  engagement_score?: number;
};

type ActionItem = {
  owner: string;
  description: string;
  due_date: string | null;
};

type ApiResponse = {
  segments: Segment[];
  engagement_by_speaker?: Record<string, number>;
  sentiment?: {
    score: number; // -1 to 1
    label: string; // "Positive" | "Neutral" | "Negative"
  };
  llm_output: {
    summary?: string[];
    action_items?: ActionItem[];
    raw_response?: string;
  };
};

type SessionEntry = {
  id: string;
  title: string;
  createdAt: string; // ISO string
  data: ApiResponse;
};

const HISTORY_KEY = "smartmeet_history_v1";
const MAX_HISTORY = 5;

// Keep a small recent-history list in localStorage so I can reopen old runs
// without re-uploading audio every time.
function loadHistory(): SessionEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(HISTORY_KEY);
    if (!raw) return [];
    return JSON.parse(raw) as SessionEntry[];
  } catch {
    return [];
  }
}

function saveHistory(entries: SessionEntry[]) {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(HISTORY_KEY, JSON.stringify(entries));
}

// Build a simple “session” object from the API response.
// I use the first summary line as a rough title, otherwise fallback.
function createSessionEntry(data: ApiResponse): SessionEntry {
  const now = new Date();
  const titleFromSummary =
    data.llm_output.summary && data.llm_output.summary.length > 0
      ? data.llm_output.summary[0]
      : "Untitled meeting";
  return {
    id: `${now.getTime()}`,
    title: titleFromSummary.slice(0, 80),
    createdAt: now.toISOString(),
    data,
  };
}

// Helper to group segments so the engagement timeline can show one row per speaker.
function groupSegmentsBySpeaker(
  segments: Segment[],
): Record<string, Segment[]> {
  const bySpeaker: Record<string, Segment[]> = {};
  segments.forEach((seg) => {
    const key = seg.speaker || "Unknown";
    if (!bySpeaker[key]) bySpeaker[key] = [];
    bySpeaker[key].push(seg);
  });
  Object.keys(bySpeaker).forEach((speaker) => {
    bySpeaker[speaker].sort((a, b) => a.start - b.start);
  });
  return bySpeaker;
}

// Tiny helpers to download the analysis as JSON or plain text.
function downloadJson(data: unknown, filename: string) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

function downloadText(text: string, filename: string) {
  const blob = new Blob([text], { type: "text/plain" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export default function HomePage() {
  // Core state: uploaded file, backend response, and simple UI flags.
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [data, setData] = useState<ApiResponse | null>(null);
  const [history, setHistory] = useState<SessionEntry[]>([]);
  const [activeSection, setActiveSection] = useState<
    "overview" | "engagement" | "summaries"
  >("overview");

  // Controls that are passed directly to the backend.
  const [llmMode, setLlmMode] = useState<"summary" | "actions" | "both">(
    "both",
  );
  const [temperature, setTemperature] = useState<number>(0.2);
  const [maxTokens, setMaxTokens] = useState<number>(512);
  const [asrModel, setAsrModel] = useState<"tiny" | "base" | "small">("base");
  const [asrLanguage, setAsrLanguage] = useState<string>("auto");
  const [maxAudioSeconds, setMaxAudioSeconds] = useState<number>(0);

  React.useEffect(() => {
    setHistory(loadHistory());
  }, []);

  // Build a plain-text export that is easy to share or paste into notes.
  const buildTextExport = () => {
    if (!data) return "";

    const lines: string[] = [];

    lines.push("SmartMeet AI – Meeting Summary");
    lines.push("");
    lines.push("=== Summary ===");
    if (data.llm_output.summary && data.llm_output.summary.length > 0) {
      data.llm_output.summary.forEach((s, i) => {
        lines.push(`${i + 1}. ${s}`);
      });
    } else {
      lines.push("No summary.");
    }

    lines.push("");
    lines.push("=== Action Items ===");
    if (
      data.llm_output.action_items &&
      data.llm_output.action_items.length > 0
    ) {
      data.llm_output.action_items.forEach((item, i) => {
        lines.push(
          `${i + 1}. [${item.owner}] ${item.description} ${
            item.due_date ? `(Due: ${item.due_date})` : ""
          }`.trim(),
        );
      });
    } else {
      lines.push("No action items.");
    }

    lines.push("");
    lines.push("=== Speakers & Engagement ===");
    if (data.engagement_by_speaker) {
      Object.entries(data.engagement_by_speaker).forEach(([speaker, score]) => {
        lines.push(`${speaker}: avg engagement ${score}/100`);
      });
    }

    lines.push("");
    lines.push("=== Transcript ===");
    data.segments.forEach((seg) => {
      lines.push(
        `[${seg.speaker}] [${seg.start.toFixed(1)}s–${seg.end.toFixed(
          1,
        )}s] ${seg.text}`,
      );
    });

    return lines.join("\n");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setData(null);
      setError(null);
    }
  };

  // Core upload + process handler: send audio + query params to FastAPI,
  // update state, and remember the run in recent history.
  const handleUpload = async () => {
    if (!file) {
      setError("Please select an audio file first.");
      return;
    }
    setLoading(true);
    setError(null);
    setData(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const params = new URLSearchParams({
        llm_mode: llmMode,
        temperature: String(temperature),
        max_tokens: String(maxTokens),
        asr_model: asrModel,
        asr_language: asrLanguage,
        max_audio_seconds: String(maxAudioSeconds),
      });

      const res = await fetch(
        `http://127.0.0.1:8000/process-meeting?${params.toString()}`,
        {
          method: "POST",
          body: formData,
        },
      );

      if (!res.ok) {
        const text = await res.text();
        throw new Error(`Backend error (${res.status}): ${text}`);
      }

      const json = (await res.json()) as ApiResponse;
      setData(json);

      const entry = createSessionEntry(json);
      setHistory((prev) => {
        const updated = [entry, ...prev].slice(0, MAX_HISTORY);
        saveHistory(updated);
        return updated;
      });
    } catch (err: any) {
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };

  const speakerCount =
    data && data.segments
      ? Array.from(new Set(data.segments.map((s) => s.speaker))).length
      : null;

  const avgEngagement =
    data && data.engagement_by_speaker
      ? Object.values(data.engagement_by_speaker).length > 0
        ? Object.values(data.engagement_by_speaker).reduce((a, b) => a + b, 0) /
          Object.values(data.engagement_by_speaker).length
        : null
      : null;

  return (
    <main className="min-h-screen bg-slate-50 text-slate-900">
      {/* Soft gradient background blobs just for a bit of visual personality */}
      <div className="absolute inset-0 pointer-events-none">
        <div className="absolute -top-24 -right-16 h-72 w-72 bg-sky-100 rounded-full blur-3xl opacity-60" />
        <div className="absolute top-40 -left-10 h-64 w-64 bg-emerald-100 rounded-full blur-3xl opacity-60" />
      </div>

      <div className="relative flex min-h-screen">
        {/* Left sidebar: navigation + recent sessions */}
        <aside className="hidden md:flex w-64 flex-col border-r border-slate-200 bg-white/90 backdrop-blur-md">
          <div className="px-6 py-5 border-b border-slate-200">
            <div className="text-xs uppercase tracking-[0.2em] text-slate-400 mb-1">
              SmartMeet AI
            </div>
            <div className="text-lg font-semibold text-sky-700">
              Meeting Intelligence
            </div>
          </div>
          <nav className="px-4 py-4 space-y-2 text-sm">
            <button
              onClick={() => setActiveSection("overview")}
              className={`w-full text-left px-3 py-2 rounded-lg font-medium flex items-center gap-2 ${
                activeSection === "overview"
                  ? "bg-sky-100 text-sky-700"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              <span className="h-2 w-2 rounded-full bg-emerald-500 animate-pulse" />
              Overview
            </button>
            <button
              onClick={() => setActiveSection("engagement")}
              className={`w-full text-left px-3 py-2 rounded-lg ${
                activeSection === "engagement"
                  ? "bg-sky-100 text-sky-700"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              Engagement
            </button>
            <button
              onClick={() => setActiveSection("summaries")}
              className={`w-full text-left px-3 py-2 rounded-lg ${
                activeSection === "summaries"
                  ? "bg-sky-100 text-sky-700"
                  : "text-slate-500 hover:bg-slate-100"
              }`}
            >
              Summaries
            </button>
          </nav>

          <div className="flex-1 px-4 pb-4 overflow-y-auto">
            <h3 className="text-xs font-semibold text-slate-500 mb-2">
              Recent Sessions
            </h3>
            {history.length === 0 ? (
              <p className="text-xs text-slate-400">
                No sessions yet. Process a meeting to save it here.
              </p>
            ) : (
              <ul className="space-y-1 text-xs">
                {history.map((entry) => (
                  <li key={entry.id}>
                    <button
                      className="w-full text-left px-2 py-1 rounded hover:bg-slate-100"
                      onClick={() => {
                        setData(entry.data);
                        setError(null);
                      }}
                    >
                      <div className="font-medium text-slate-700 truncate">
                        {entry.title}
                      </div>
                      <div className="text-[10px] text-slate-500">
                        {new Date(entry.createdAt).toLocaleString()}
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>

          <div className="px-6 py-4 text-xs text-slate-400 border-t border-slate-200">
            Built for B.Tech AI &amp; ML · 2025–26
          </div>
        </aside>

        {/* Main content area */}
        <div className="flex-1 flex flex-col">
          {/* Top header with tagline */}
          <header className="border-b border-slate-200 bg-white/80 backdrop-blur-md">
            <div className="max-w-6xl mx-auto px-4 lg:px-8 py-4 flex items-center justify-between">
              <div>
                <h1 className="text-xl md:text-2xl font-semibold text-slate-900">
                  SmartMeet – Meeting Analytics Dashboard
                </h1>
                <p className="text-xs md:text-sm text-slate-500">
                  Upload a recorded meeting to analyze speakers, engagement, and
                  action items.
                </p>
              </div>
              <div className="hidden md:flex items-center gap-3 text-xs text-slate-500">
                <span className="px-2 py-1 rounded-full bg-emerald-50 border border-emerald-200 text-emerald-700">
                  Local · Privacy-first
                </span>
              </div>
            </div>
          </header>

          {/* Main dashboard layout */}
          <div className="flex-1">
            <div className="max-w-6xl mx-auto px-4 lg:px-8 py-6 space-y-6">
              {/* High-level stats at the top */}
              <div className="grid gap-3 md:grid-cols-4">
                <div className="rounded-2xl border border-slate-200 bg-white/80 backdrop-blur-sm px-4 py-3 flex items-center justify-between">
                  <div>
                    <p className="text-xs uppercase tracking-wide text-slate-400">
                      Last Meeting
                    </p>
                    <p className="text-sm font-semibold text-slate-900">
                      {data ? "Processed" : "Not processed"}
                    </p>
                  </div>
                  <span className="inline-flex h-8 w-8 items-center justify-center rounded-full bg-sky-100 text-sky-600 text-sm font-semibold">
                    {data ? "✓" : "–"}
                  </span>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white/80 backdrop-blur-sm px-4 py-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">
                    Speakers
                  </p>
                  <p className="text-sm font-semibold text-slate-900">
                    {data && speakerCount !== null ? speakerCount : "—"}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white/80 backdrop-blur-sm px-4 py-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">
                    Avg Engagement
                  </p>
                  <p className="text-sm font-semibold text-slate-900">
                    {avgEngagement !== null ? avgEngagement.toFixed(1) : "—"}
                  </p>
                </div>

                <div className="rounded-2xl border border-slate-200 bg-white/80 backdrop-blur-sm px-4 py-3">
                  <p className="text-xs uppercase tracking-wide text-slate-400">
                    Meeting Sentiment
                  </p>
                  {data && data.sentiment ? (
                    <div className="space-y-1">
                      <p className="text-sm font-semibold text-slate-900">
                        {data.sentiment.label}{" "}
                        <span className="text-xs text-slate-500">
                          ({data.sentiment.score})
                        </span>
                      </p>
                      {/* Horizontal sentiment bar: marker slides from -1 (left, red) to +1 (right, green) */}
                      <div className="h-2 w-full bg-slate-100 rounded-full overflow-hidden">
                        <div className="relative h-full">
                          {/* Neutral center line at 0 */}
                          <div className="absolute left-1/2 top-0 bottom-0 w-px bg-slate-300" />
                          {/* Position marker */}
                          {(() => {
                            const score = Math.max(
                              -1,
                              Math.min(1, data.sentiment!.score),
                            );
                            const percent = ((score + 1) / 2) * 100; // -1 -> 0, 0 -> 50, 1 -> 100
                            const color =
                              score > 0.1
                                ? "bg-emerald-500"
                                : score < -0.1
                                  ? "bg-rose-500"
                                  : "bg-slate-400";
                            return (
                              <div
                                className={`h-full ${color}`}
                                style={{
                                  width: "6%",
                                  transform: `translateX(${percent - 3}%)`,
                                }}
                              />
                            );
                          })()}
                        </div>
                      </div>
                    </div>
                  ) : (
                    <p className="text-sm font-semibold text-slate-900">—</p>
                  )}
                </div>
              </div>

              {/* Upload card + results grid */}
              <div className="grid gap-6 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1.3fr)]">
                <div className="space-y-4">
                  {/* Upload + settings */}
                  <div className="rounded-2xl border border-slate-200 bg-white shadow-[0_12px_30px_rgba(15,23,42,0.08)] p-4 md:p-5">
                    <h2 className="text-base md:text-lg font-semibold mb-1 text-slate-900">
                      Upload Meeting Audio
                    </h2>
                    <p className="text-xs md:text-sm text-slate-500 mb-4">
                      Supported formats: WAV, MP3. Duration up to ~60 minutes
                      for best performance.
                    </p>

                    <div className="space-y-3">
                      <input
                        type="file"
                        accept="audio/*"
                        onChange={handleFileChange}
                        className="block w-full text-sm text-slate-800"
                      />

                      {/* ASR controls (Whisper settings) */}
                      <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
                        <p className="text-xs font-semibold text-slate-600">
                          ASR Settings (Whisper)
                        </p>
                        <div className="flex flex-wrap gap-2 text-xs">
                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">Model:</span>
                            <select
                              value={asrModel}
                              onChange={(e) =>
                                setAsrModel(
                                  e.target.value as "tiny" | "base" | "small",
                                )
                              }
                              className="border border-slate-300 rounded px-1 py-0.5 text-xs"
                            >
                              <option value="tiny">
                                tiny (fastest, lower accuracy)
                              </option>
                              <option value="base">base (balanced)</option>
                              <option value="small">
                                small (slower, better accuracy)
                              </option>
                            </select>
                          </label>

                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">Language:</span>
                            <select
                              value={asrLanguage}
                              onChange={(e) => setAsrLanguage(e.target.value)}
                              className="border border-slate-300 rounded px-1 py-0.5 text-xs"
                            >
                              <option value="auto">Auto-detect</option>
                              <option value="en">English</option>
                              <option value="hi">Hindi</option>
                              <option value="mr">Marathi</option>
                              {/* add more ISO codes if you want */}
                            </select>
                          </label>

                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">
                              Fast demo (seconds):
                            </span>
                            <input
                              type="number"
                              min={0}
                              max={600}
                              step={30}
                              value={maxAudioSeconds}
                              onChange={(e) =>
                                setMaxAudioSeconds(
                                  Math.min(
                                    600,
                                    Math.max(
                                      0,
                                      Number.isNaN(parseInt(e.target.value))
                                        ? 0
                                        : parseInt(e.target.value, 10),
                                    ),
                                  ),
                                )
                              }
                              className="w-20 border border-slate-300 rounded px-1 py-0.5 text-xs"
                            />
                          </label>
                        </div>
                      </div>

                      {/* LLM parameter controls */}
                      <div className="border border-slate-200 rounded-lg p-3 bg-slate-50 space-y-2">
                        <p className="text-xs font-semibold text-slate-600">
                          LLM Settings
                        </p>

                        <div className="flex flex-wrap gap-2 text-xs">
                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">Mode:</span>
                            <select
                              value={llmMode}
                              onChange={(e) =>
                                setLlmMode(
                                  e.target.value as
                                    | "summary"
                                    | "actions"
                                    | "both",
                                )
                              }
                              className="border border-slate-300 rounded px-1 py-0.5 text-xs"
                            >
                              <option value="both">Summary + Actions</option>
                              <option value="summary">Summary only</option>
                              <option value="actions">Action items only</option>
                            </select>
                          </label>

                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">Temp:</span>
                            <input
                              type="number"
                              min={0}
                              max={1}
                              step={0.1}
                              value={temperature}
                              onChange={(e) =>
                                setTemperature(
                                  Math.min(
                                    1,
                                    Math.max(
                                      0,
                                      Number.isNaN(parseFloat(e.target.value))
                                        ? 0.2
                                        : parseFloat(e.target.value),
                                    ),
                                  ),
                                )
                              }
                              className="w-16 border border-slate-300 rounded px-1 py-0.5 text-xs"
                            />
                          </label>

                          <label className="flex items-center gap-1">
                            <span className="text-slate-600">Max tokens:</span>
                            <input
                              type="number"
                              min={64}
                              max={2048}
                              step={64}
                              value={maxTokens}
                              onChange={(e) =>
                                setMaxTokens(
                                  Math.min(
                                    2048,
                                    Math.max(
                                      64,
                                      Number.isNaN(parseInt(e.target.value))
                                        ? 512
                                        : parseInt(e.target.value, 10),
                                    ),
                                  ),
                                )
                              }
                              className="w-20 border border-slate-300 rounded px-1 py-0.5 text-xs"
                            />
                          </label>
                        </div>
                      </div>

                      <button
                        onClick={handleUpload}
                        disabled={loading || !file}
                        className="px-4 py-2 bg-sky-600 hover:bg-sky-700 disabled:bg-slate-300 disabled:text-slate-500 rounded text-sm font-semibold text-white"
                      >
                        {loading ? "Processing..." : "Upload & Process"}
                      </button>
                      {error && <p className="text-red-600 text-sm">{error}</p>}
                    </div>
                  </div>

                  {/* Left side: transcript and engagement visuals */}
                  {data && activeSection !== "summaries" && (
                    <div className="rounded-2xl border border-slate-200 bg-white shadow-[0_12px_30px_rgba(15,23,42,0.08)] p-4 md:p-5 overflow-y-auto max-h-[70vh]">
                      <h2 className="text-xl font-semibold mb-2 text-slate-900">
                        Transcript (Speaker View)
                      </h2>

                      {data && data.sentiment && (
                        <div className="mb-2 text-xs text-slate-600">
                          Overall sentiment:{" "}
                          <span
                            className={
                              data.sentiment.label === "Positive"
                                ? "text-emerald-600"
                                : data.sentiment.label === "Negative"
                                  ? "text-rose-600"
                                  : "text-slate-600"
                            }
                          >
                            {data.sentiment.label} ({data.sentiment.score})
                          </span>
                        </div>
                      )}

                      {data.engagement_by_speaker && (
                        <div className="mb-4">
                          <h3 className="text-lg font-semibold text-sky-700 mb-1">
                            Avg Engagement by Speaker
                          </h3>
                          <div className="space-y-1 text-sm">
                            {Object.entries(data.engagement_by_speaker).map(
                              ([speaker, score]) => (
                                <div
                                  key={speaker}
                                  className="flex items-center gap-2"
                                >
                                  <span className="w-24 text-sky-700">
                                    {speaker}
                                  </span>
                                  <div className="flex-1 bg-slate-100 rounded h-3 overflow-hidden">
                                    <div
                                      className="h-3 bg-emerald-500"
                                      style={{
                                        width: `${Math.min(score, 100)}%`,
                                      }}
                                    />
                                  </div>
                                  <span className="w-12 text-right text-emerald-700 text-xs">
                                    {score}
                                  </span>
                                </div>
                              ),
                            )}
                          </div>
                        </div>
                      )}

                      <div className="mb-4">
                        <h3 className="text-lg font-semibold text-sky-700 mb-1">
                          Engagement Timeline
                        </h3>
                        <div className="space-y-2 text-xs">
                          {Object.entries(
                            groupSegmentsBySpeaker(data.segments),
                          ).map(([speaker, segs]) => (
                            <div key={speaker}>
                              <div className="mb-1 text-sky-700">
                                {speaker}
                              </div>
                              <div className="flex gap-1">
                                {segs.map((seg, idx) => {
                                  const score = seg.engagement_score ?? 0;
                                  const height =
                                    10 +
                                    (Math.min(score, 100) / 100) * 30;
                                  return (
                                    <div
                                      key={idx}
                                      className="flex-1 bg-slate-100 rounded flex items-end justify-center"
                                      title={`[${seg.start.toFixed(
                                        1,
                                      )}s–${seg.end.toFixed(
                                        1,
                                      )}s] Engagement: ${score}`}
                                    >
                                      <div
                                        className="w-full bg-emerald-500"
                                        style={{ height: `${height}px` }}
                                      />
                                    </div>
                                  );
                                })}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div className="space-y-2 text-sm">
                        {data.segments.map((seg, idx) => (
                          <div key={idx}>
                            <span className="inline-flex items-center justify-center mr-2 h-6 w-6 rounded-full bg-sky-50 text-sky-700 text-xs font-semibold">
                              {seg.speaker.replace("Speaker ", "S")}
                            </span>
                            <span className="text-slate-600 text-xs">
                              [{seg.start.toFixed(1)}s – {seg.end.toFixed(1)}s]
                            </span>
                            {seg.engagement_score !== undefined && (
                              <span className="inline-flex items-center rounded-full bg-emerald-50 text-emerald-700 text-[11px] font-medium px-2 py-0.5 ml-2">
                                {seg.engagement_score} / 100
                              </span>
                            )}
                            <div className="mt-0.5">{seg.text}</div>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>

                {/* Right-hand summary + actions card */}
                {activeSection !== "engagement" && (
                  <div className="rounded-2xl border border-slate-200 bg-white shadow-[0_12px_30px_rgba(15,23,42,0.08)] p-4 md:p-5 overflow-y-auto max-h-[70vh]">
                    <div className="flex items-center justify-between mb-2">
                      <h2 className="text-xl font-semibold text-slate-900">
                        Summary &amp; Action Items
                      </h2>
                      {data && (
                        <div className="flex gap-2">
                          <button
                            className="px-2 py-1 text-xs rounded border border-slate-300 text-slate-700 hover:bg-slate-100"
                            onClick={() => {
                              const filename = `smartmeet_session_${Date.now()}.json`;
                              downloadJson(data, filename);
                            }}
                          >
                            Download JSON
                          </button>
                          <button
                            className="px-2 py-1 text-xs rounded border border-slate-300 text-slate-700 hover:bg-slate-100"
                            onClick={() => {
                              const text = buildTextExport();
                              if (!text) return;
                              const filename = `smartmeet_summary_${Date.now()}.txt`;
                              downloadText(text, filename);
                            }}
                          >
                            Download TXT
                          </button>
                        </div>
                      )}
                    </div>

                    {data ? (
                      <>
                        <div className="mb-4">
                          <h3 className="text-lg font-semibold text-sky-700 mb-1">
                            Summary
                          </h3>
                          {data.llm_output.summary &&
                          data.llm_output.summary.length > 0 ? (
                            <ul className="list-disc list-inside text-sm space-y-1">
                              {data.llm_output.summary.map((point, idx) => (
                                <li key={idx}>{point}</li>
                              ))}
                            </ul>
                          ) : (
                            <p className="text-sm text-slate-500">
                              No summary returned.
                            </p>
                          )}
                        </div>

                        <div>
                          <h3 className="text-lg font-semibold text-sky-700 mb-1">
                            Action Items
                          </h3>
                          {data.llm_output.action_items &&
                          data.llm_output.action_items.length > 0 ? (
                            <table className="w-full text-sm border-separate border-spacing-y-1">
                              <thead>
                                <tr className="text-xs text-slate-500">
                                  <th className="text-left py-1 px-2">Owner</th>
                                  <th className="text-left py-1 px-2">
                                    Description
                                  </th>
                                  <th className="text-left py-1 px-2">
                                    Due Date
                                  </th>
                                </tr>
                              </thead>
                              <tbody>
                                {data.llm_output.action_items.map(
                                  (item, idx) => (
                                    <tr
                                      key={idx}
                                      className="bg-slate-50 hover:bg-slate-100 transition-colors"
                                    >
                                      <td className="py-2 px-2 rounded-l-xl text-sky-700 font-medium">
                                        {item.owner}
                                      </td>
                                      <td className="py-2 px-2">
                                        {item.description}
                                      </td>
                                      <td className="py-2 px-2 rounded-r-xl text-slate-600 text-xs">
                                        {item.due_date || "Not specified"}
                                      </td>
                                    </tr>
                                  ),
                                )}
                              </tbody>
                            </table>
                          ) : (
                            <p className="text-sm text-slate-500">
                              No action items found.
                            </p>
                          )}
                        </div>
                      </>
                    ) : (
                      <p className="text-sm text-slate-500">
                        Upload and process a meeting to see summaries and action
                        items here.
                      </p>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}