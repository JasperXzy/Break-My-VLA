import { useRef, useState } from 'react'
import {
  CheckCircle2,
  Film,
  Flame,
  Pause,
  Play,
  RotateCcw,
  XCircle,
} from 'lucide-react'
import type { ReplayComparison } from '../types/benchmark'

interface DualGifReplayPlayerProps {
  replays: ReplayComparison[]
}

export function DualGifReplayPlayer({ replays }: DualGifReplayPlayerProps) {
  const [activeIndex, setActiveIndex] = useState(0)
  const [syncSeed, setSyncSeed] = useState(1)
  const [isFrozen, setIsFrozen] = useState(false)

  const leftImgRef = useRef<HTMLImageElement | null>(null)
  const rightImgRef = useRef<HTMLImageElement | null>(null)
  const leftFreezeRef = useRef<HTMLCanvasElement | null>(null)
  const rightFreezeRef = useRef<HTMLCanvasElement | null>(null)

  if (replays.length === 0) {
    return (
      <div className="arcade-panel rounded-xl p-4 font-mono text-xs text-slate-400">
        No replay GIF comparisons included in this benchmark JSON.
      </div>
    )
  }

  const current = replays[Math.min(activeIndex, replays.length - 1)]
  const speedup =
    current.degraded_latency_ms > 0
      ? (current.baseline_latency_ms / current.degraded_latency_ms).toFixed(1)
      : '1.0'

  const withSyncQuery = (src: string) => {
    const sep = src.includes('?') ? '&' : '?'
    return `${src}${sep}sync=${syncSeed}`
  }

  const handleRestartBoth = () => {
    setIsFrozen(false)
    setSyncSeed((s) => s + 1)
  }

  const handleToggleFreeze = () => {
    if (!isFrozen) {
      const freezeFrame = (
        img: HTMLImageElement | null,
        canvas: HTMLCanvasElement | null,
      ) => {
        if (!img || !canvas || img.naturalWidth === 0) return
        canvas.width = img.naturalWidth
        canvas.height = img.naturalHeight
        const ctx = canvas.getContext('2d')
        ctx?.drawImage(img, 0, 0)
      }
      freezeFrame(leftImgRef.current, leftFreezeRef.current)
      freezeFrame(rightImgRef.current, rightFreezeRef.current)
      setIsFrozen(true)
    } else {
      setIsFrozen(false)
    }
  }

  return (
    <section className="arcade-panel rounded-xl p-3.5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 font-mono text-xs font-bold tracking-wider text-fuchsia-300">
            <Film className="h-4 w-4" />
            <span>DUAL-SCREEN GIF REPLAY COMPARISON // 破防双屏回放馆</span>
          </div>
          <p className="font-mono text-[10px] text-slate-400">
            左侧基准精度 vs 右侧量化/扰动对比，支持无 GPU 访客在 GitHub Pages 直接同步回放
          </p>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={handleToggleFreeze}
            className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-slate-200 transition hover:border-cyan-400/60 hover:text-white active:scale-95"
          >
            {isFrozen ? <Play className="h-3.5 w-3.5 text-emerald-300" /> : <Pause className="h-3.5 w-3.5 text-amber-300" />}
            <span>{isFrozen ? 'RESUME GIF' : 'FREEZE FRAME'}</span>
          </button>
          <button
            type="button"
            onClick={handleRestartBoth}
            className="flex items-center gap-1.5 rounded-lg border border-fuchsia-400/50 bg-fuchsia-500/15 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-fuchsia-200 transition hover:bg-fuchsia-500/25 active:scale-95"
          >
            <RotateCcw className="h-3.5 w-3.5" />
            <span>SYNC RESTART</span>
          </button>
        </div>
      </div>

      {/* Scenario selector tabs */}
      <div className="mb-3 flex flex-wrap gap-1.5">
        {replays.map((item, idx) => {
          const active = idx === activeIndex
          return (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setActiveIndex(idx)
                setIsFrozen(false)
                setSyncSeed((s) => s + 1)
              }}
              className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] transition ${
                active
                  ? 'border-fuchsia-400/70 bg-fuchsia-500/20 font-semibold text-white shadow-[0_0_14px_rgba(232,121,249,0.25)]'
                  : 'border-slate-700/80 bg-slate-900/70 text-slate-400 hover:border-slate-500 hover:text-slate-200'
              }`}
            >
              <Flame className="h-3 w-3 text-rose-400" />
              <span>{item.title}</span>
            </button>
          )
        })}
      </div>

      {/* Dual-Screen GIF Viewport */}
      <div className="grid grid-cols-1 gap-3 md:grid-cols-2">
        {/* Left: Baseline */}
        <div className="overflow-hidden rounded-xl border border-emerald-500/35 bg-slate-950/90">
          <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1.5 font-mono text-[11px]">
            <div className="flex items-center gap-2">
              <span className="rounded bg-emerald-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase text-emerald-300">
                {current.baseline_precision}
              </span>
              <span className="font-semibold text-slate-100">{current.baseline_label}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-slate-400">
                {current.baseline_latency_ms.toFixed(1)} ms
              </span>
              <span
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                  current.baseline_outcome === 'pass'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-rose-500/20 text-rose-300'
                }`}
              >
                {current.baseline_outcome === 'pass' ? (
                  <CheckCircle2 className="h-3 w-3" />
                ) : (
                  <XCircle className="h-3 w-3" />
                )}
                {current.baseline_outcome}
              </span>
            </div>
          </div>
          <div className="relative aspect-[16/10] w-full bg-slate-950">
            <img
              ref={leftImgRef}
              src={withSyncQuery(current.baseline_gif)}
              alt={current.baseline_label}
              className={`h-full w-full object-cover ${isFrozen ? 'opacity-0' : 'opacity-100'}`}
            />
            <canvas
              ref={leftFreezeRef}
              className={`absolute inset-0 h-full w-full object-cover ${isFrozen ? 'block' : 'hidden'}`}
            />
            <div className="crt-scanlines pointer-events-none absolute inset-0 opacity-30" />
          </div>
        </div>

        {/* Right: Degraded / Chaos */}
        <div className="overflow-hidden rounded-xl border border-rose-500/40 bg-slate-950/90">
          <div className="flex items-center justify-between border-b border-slate-800 px-3 py-1.5 font-mono text-[11px]">
            <div className="flex items-center gap-2">
              <span className="rounded bg-fuchsia-500/20 px-1.5 py-0.5 text-[10px] font-bold uppercase text-fuchsia-300">
                {current.degraded_precision}
              </span>
              <span className="font-semibold text-slate-100">{current.degraded_label}</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-[10px] text-cyan-300">
                {current.degraded_latency_ms.toFixed(1)} ms ({speedup}x)
              </span>
              <span
                className={`inline-flex items-center gap-1 rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${
                  current.degraded_outcome === 'pass'
                    ? 'bg-emerald-500/20 text-emerald-300'
                    : 'bg-rose-500/20 text-rose-300'
                }`}
              >
                {current.degraded_outcome === 'pass' ? (
                  <CheckCircle2 className="h-3 w-3" />
                ) : (
                  <XCircle className="h-3 w-3" />
                )}
                {current.degraded_outcome === 'fail' ? 'BROKE' : 'PASS'}
              </span>
            </div>
          </div>
          <div className="relative aspect-[16/10] w-full bg-slate-950">
            <img
              ref={rightImgRef}
              src={withSyncQuery(current.degraded_gif)}
              alt={current.degraded_label}
              className={`h-full w-full object-cover ${isFrozen ? 'opacity-0' : 'opacity-100'}`}
            />
            <canvas
              ref={rightFreezeRef}
              className={`absolute inset-0 h-full w-full object-cover ${isFrozen ? 'block' : 'hidden'}`}
            />
            <div className="crt-scanlines pointer-events-none absolute inset-0 opacity-30" />
          </div>
        </div>
      </div>

      {/* Post-mortem annotation bar */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2 font-mono text-[11px]">
        <div className="flex flex-wrap items-center gap-2">
          <span className="rounded border border-rose-500/40 bg-rose-500/15 px-1.5 py-0.5 text-[10px] font-bold uppercase text-rose-300">
            CHAOS: {current.chaos_tag}
          </span>
          <span className="text-slate-300">{current.failure_note}</span>
        </div>
        <span className="text-[10px] text-slate-500">PROMPT: {current.prompt}</span>
      </div>
    </section>
  )
}
