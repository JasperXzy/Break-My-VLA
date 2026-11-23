import { useState, type FormEvent } from 'react'
import {
  Activity,
  Cpu,
  Download,
  Loader2,
  Pause,
  Play,
  Radio,
  RotateCcw,
  Send,
  Settings2,
  Sparkles,
  Wifi,
  WifiOff,
  Zap,
} from 'lucide-react'
import {
  type ConnectionMode,
  type ConnectionState,
  type FrameStatus,
  PRESET_PROMPTS,
} from '../types/protocol'

interface TopStatusBarProps {
  wsUrl: string
  mode: ConnectionMode
  connectionState: ConnectionState
  modelName: string
  fps: number
  inferMs: number
  step: number
  status: FrameStatus
  prompt: string
  isExportingGif: boolean
  onConnectLive: (url: string) => void
  onSwitchToDemo: () => void
  onSubmitPrompt: (prompt: string) => void
  onReset: () => void
  onTogglePause: () => void
  onExportGif: () => void
}

export function TopStatusBar({
  wsUrl,
  mode,
  connectionState,
  modelName,
  fps,
  inferMs,
  step,
  status,
  prompt,
  isExportingGif,
  onConnectLive,
  onSwitchToDemo,
  onSubmitPrompt,
  onReset,
  onTogglePause,
  onExportGif,
}: TopStatusBarProps) {
  const [promptDraft, setPromptDraft] = useState(prompt)
  const [wsDraft, setWsDraft] = useState(wsUrl)
  const [showWsPopover, setShowWsPopover] = useState(false)
  const [showPresets, setShowPresets] = useState(false)

  const isPaused = status === 'paused'

  const handlePromptSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (promptDraft.trim()) {
      onSubmitPrompt(promptDraft.trim())
    }
  }

  const handleWsSubmit = (e: FormEvent) => {
    e.preventDefault()
    onConnectLive(wsDraft)
    setShowWsPopover(false)
  }

  const latencyColor =
    inferMs > 120 ? 'text-rose-400' : inferMs > 55 ? 'text-amber-300' : 'text-emerald-300'

  return (
    <header className="arcade-panel relative z-30 border-b border-slate-700/80 px-3 py-2.5 lg:px-4">
      <div className="flex flex-wrap items-center justify-between gap-2.5">
        {/* Brand + Connection Mode + Model Telemetry */}
        <div className="flex flex-wrap items-center gap-2.5">
          {/* Logo */}
          <div className="flex items-center gap-2 pr-1">
            <div className="relative flex h-9 w-9 items-center justify-center rounded-lg border border-cyan-400/50 bg-cyan-500/10 shadow-[0_0_18px_rgba(34,211,238,0.28)]">
              <Zap className="h-5 w-5 text-cyan-300" />
              <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full bg-fuchsia-400 shadow-[0_0_8px_#e879f9]" />
            </div>
            <div className="leading-tight">
              <div className="flex items-center gap-1.5">
                <h1 className="font-mono text-sm font-bold tracking-wider text-white sm:text-base">
                  BREAK-MY-VLA
                </h1>
                <span className="hidden rounded border border-fuchsia-500/40 bg-fuchsia-500/15 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-fuchsia-300 xl:inline">
                  ARCADE v1
                </span>
              </div>
              <p className="font-mono text-[10px] tracking-widest text-slate-400">
                ADVERSARIAL VLA TELEMETRY LAB
              </p>
            </div>
          </div>

          <div className="hidden h-7 w-px bg-slate-700/70 sm:block" />

          {/* Connection Status LED & Endpoint Switcher */}
          <div className="relative">
            <button
              type="button"
              onClick={() => {
                setWsDraft(wsUrl)
                setShowWsPopover((prev) => !prev)
              }}
              className={`flex items-center gap-2 rounded-lg border px-2.5 py-1.5 font-mono text-xs font-semibold transition ${
                mode === 'live'
                  ? 'border-emerald-400/50 bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/20'
                  : connectionState === 'connecting'
                    ? 'border-cyan-400/50 bg-cyan-500/15 text-cyan-300'
                    : 'border-amber-400/50 bg-amber-500/15 text-amber-300 hover:bg-amber-500/20'
              }`}
              title="Click to configure backend WebSocket endpoint or switch mode"
            >
              <span className="relative flex h-2.5 w-2.5">
                <span
                  className={`absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 ${
                    mode === 'live'
                      ? 'bg-emerald-400'
                      : connectionState === 'connecting'
                        ? 'bg-cyan-400'
                        : 'bg-amber-400'
                  }`}
                />
                <span
                  className={`relative inline-flex h-2.5 w-2.5 rounded-full ${
                    mode === 'live'
                      ? 'bg-emerald-400'
                      : connectionState === 'connecting'
                        ? 'bg-cyan-400'
                        : 'bg-amber-400'
                  }`}
                />
              </span>
              {mode === 'live' ? (
                <Wifi className="h-3.5 w-3.5" />
              ) : (
                <WifiOff className="h-3.5 w-3.5" />
              )}
              <span>
                {connectionState === 'connecting'
                  ? 'SYNCING...'
                  : mode === 'live'
                    ? 'LIVE'
                    : 'DEMO'}
              </span>
              <Settings2 className="h-3 w-3 opacity-70" />
            </button>

            {showWsPopover && (
              <div className="arcade-panel absolute left-0 top-11 z-50 w-80 rounded-xl border border-cyan-500/40 p-3 shadow-2xl">
                <div className="mb-2 flex items-center justify-between">
                  <span className="font-mono text-[11px] font-semibold uppercase tracking-wider text-cyan-300">
                    Backend WS Endpoint
                  </span>
                  <span className="font-mono text-[10px] text-slate-400">
                    {mode === 'live' ? 'Live Socket' : 'Local Fallback'}
                  </span>
                </div>
                <form onSubmit={handleWsSubmit} className="space-y-2.5">
                  <input
                    type="text"
                    value={wsDraft}
                    onChange={(e) => setWsDraft(e.target.value)}
                    placeholder="ws://localhost:8000/ws/session"
                    className="w-full rounded-lg border border-slate-700 bg-slate-950/90 px-2.5 py-1.5 font-mono text-xs text-slate-100 outline-none focus:border-cyan-400"
                  />
                  <div className="flex items-center gap-2">
                    <button
                      type="submit"
                      className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-emerald-400/50 bg-emerald-500/20 px-2.5 py-1.5 font-mono text-xs font-semibold text-emerald-200 transition hover:bg-emerald-500/30"
                    >
                      <Radio className="h-3.5 w-3.5" />
                      Connect Live
                    </button>
                    <button
                      type="button"
                      onClick={() => {
                        onSwitchToDemo()
                        setShowWsPopover(false)
                      }}
                      className="flex flex-1 items-center justify-center gap-1.5 rounded-lg border border-amber-400/50 bg-amber-500/15 px-2.5 py-1.5 font-mono text-xs font-semibold text-amber-200 transition hover:bg-amber-500/25"
                    >
                      Local Demo
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>

          {/* Model + Telemetry Chips */}
          <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs">
            <div className="flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/80 px-2.5 py-1.5 text-slate-200">
              <Cpu className="h-3.5 w-3.5 text-cyan-400" />
              <span className="max-w-[160px] truncate font-medium">{modelName}</span>
            </div>
            <div className="flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/80 px-2.5 py-1.5">
              <Activity className="h-3.5 w-3.5 text-emerald-400" />
              <span className="font-semibold text-white">{fps.toFixed(1)}</span>
              <span className="text-[10px] text-slate-400">FPS</span>
            </div>
            <div className="flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-900/80 px-2.5 py-1.5">
              <span className={`font-semibold ${latencyColor}`}>{inferMs.toFixed(1)}</span>
              <span className="text-[10px] text-slate-400">ms</span>
            </div>
            <div className="hidden items-center gap-1 rounded-lg border border-slate-800 bg-slate-950/70 px-2 py-1.5 text-[11px] text-slate-400 2xl:flex">
              <span>STEP</span>
              <span className="font-semibold text-slate-200">{String(step).padStart(4, '0')}</span>
            </div>
          </div>
        </div>

        {/* Task Prompt Input + Action Controls */}
        <div className="flex flex-1 flex-wrap items-center justify-end gap-2">
          <form
            onSubmit={handlePromptSubmit}
            className="relative flex min-w-[230px] max-w-md flex-1 items-center"
          >
            <button
              type="button"
              onClick={() => setShowPresets((p) => !p)}
              title="Preset task prompts"
              className="absolute left-2 rounded p-1 text-cyan-400/80 transition hover:bg-slate-800 hover:text-cyan-300"
            >
              <Sparkles className="h-3.5 w-3.5" />
            </button>
            <input
              type="text"
              value={promptDraft}
              onChange={(e) => setPromptDraft(e.target.value)}
              placeholder="Enter VLA task instruction..."
              className="w-full rounded-lg border border-slate-700/90 bg-slate-950/90 py-1.5 pl-8 pr-16 font-mono text-xs text-slate-100 placeholder-slate-500 outline-none transition focus:border-cyan-400/80 focus:ring-1 focus:ring-cyan-400/40"
            />
            <button
              type="submit"
              className="absolute right-1.5 flex items-center gap-1 rounded-md border border-cyan-500/40 bg-cyan-500/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-cyan-200 transition hover:bg-cyan-500/25"
            >
              <Send className="h-2.5 w-2.5" />
              SET
            </button>

            {showPresets && (
              <div className="arcade-panel absolute left-0 top-10 z-50 w-full rounded-xl border border-slate-700 p-1.5 shadow-2xl">
                <div className="px-2 py-1 font-mono text-[10px] uppercase tracking-wider text-slate-400">
                  Quick LIBERO Prompts
                </div>
                {PRESET_PROMPTS.map((item) => (
                  <button
                    key={item}
                    type="button"
                    onClick={() => {
                      setPromptDraft(item)
                      onSubmitPrompt(item)
                      setShowPresets(false)
                    }}
                    className="block w-full rounded-lg px-2 py-1.5 text-left font-mono text-[11px] text-slate-200 transition hover:bg-cyan-500/15 hover:text-cyan-200"
                  >
                    {item}
                  </button>
                ))}
              </div>
            )}
          </form>

          {/* Controls: Reset / Pause / Export GIF */}
          <div className="flex items-center gap-1.5">
            <button
              type="button"
              onClick={onReset}
              className="flex items-center gap-1.5 rounded-lg border border-slate-600/80 bg-slate-800/80 px-2.5 py-1.5 font-mono text-xs font-semibold text-slate-200 transition hover:border-cyan-400/60 hover:bg-slate-800 hover:text-white active:scale-95"
              title="Reset episode (Shortcut: R)"
            >
              <RotateCcw className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">RESET</span>
            </button>

            <button
              type="button"
              onClick={onTogglePause}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-xs font-semibold transition active:scale-95 ${
                isPaused
                  ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200 hover:bg-emerald-500/30'
                  : 'border-amber-400/50 bg-amber-500/15 text-amber-200 hover:bg-amber-500/25'
              }`}
              title="Pause / Resume simulation (Shortcut: Space)"
            >
              {isPaused ? <Play className="h-3.5 w-3.5" /> : <Pause className="h-3.5 w-3.5" />}
              <span className="hidden sm:inline">{isPaused ? 'RESUME' : 'PAUSE'}</span>
            </button>

            <button
              type="button"
              onClick={onExportGif}
              disabled={isExportingGif}
              className="flex items-center gap-1.5 rounded-lg border border-fuchsia-400/50 bg-fuchsia-500/15 px-2.5 py-1.5 font-mono text-xs font-semibold text-fuchsia-200 transition hover:bg-fuchsia-500/25 active:scale-95 disabled:opacity-60"
              title="Export animated GIF replay"
            >
              {isExportingGif ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
              ) : (
                <Download className="h-3.5 w-3.5" />
              )}
              <span className="hidden md:inline">EXPORT GIF</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  )
}
