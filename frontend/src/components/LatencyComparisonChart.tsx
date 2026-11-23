import { useMemo, useState } from 'react'
import { ArrowUpDown, Gauge } from 'lucide-react'
import type { BenchmarkEntry } from '../types/benchmark'
import type { Precision } from '../types/protocol'

interface LatencyComparisonChartProps {
  entries: BenchmarkEntry[]
  selectedIds: string[]
  onToggleSelect: (id: string) => void
}

type SortKey = 'latency' | 'score' | 'vram'

const PRECISION_BAR_CLASS: Record<Precision, string> = {
  bf16: 'from-emerald-500 to-teal-400',
  fp8: 'from-cyan-500 to-sky-400',
  int8: 'from-amber-500 to-yellow-400',
  nvfp4: 'from-fuchsia-500 to-pink-400',
}

const PRECISION_BADGE_CLASS: Record<Precision, string> = {
  bf16: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-300',
  fp8: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-300',
  int8: 'border-amber-400/50 bg-amber-500/15 text-amber-300',
  nvfp4: 'border-fuchsia-400/50 bg-fuchsia-500/15 text-fuchsia-300',
}

export function LatencyComparisonChart({
  entries,
  selectedIds,
  onToggleSelect,
}: LatencyComparisonChartProps) {
  const [sortBy, setSortBy] = useState<SortKey>('latency')

  const sorted = useMemo(() => {
    const copy = [...entries]
    if (sortBy === 'latency') {
      copy.sort((a, b) => a.latency.mean_ms - b.latency.mean_ms)
    } else if (sortBy === 'score') {
      copy.sort((a, b) => b.overall_score - a.overall_score)
    } else {
      copy.sort((a, b) => a.vram_gb - b.vram_gb)
    }
    return copy
  }, [entries, sortBy])

  const maxMs = Math.max(85, ...entries.map((e) => e.latency.p95_ms))
  const budget20HzPct = Math.min(100, (50 / maxMs) * 100)

  return (
    <div className="arcade-panel flex h-full flex-col rounded-xl p-3.5">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 font-mono text-xs font-bold tracking-wider text-amber-300">
            <Gauge className="h-4 w-4" />
            <span>LATENCY & THROUGHPUT COMPARISON</span>
          </div>
          <p className="font-mono text-[10px] text-slate-400">
            延迟对比条（Mean 实条 + P95 刻度线 · 虚线为 50ms / 20Hz 实时红线）
          </p>
        </div>

        <div className="flex items-center gap-1 rounded-lg border border-slate-700/80 bg-slate-950/80 p-0.5 font-mono text-[10px]">
          <ArrowUpDown className="mx-1 h-3 w-3 text-slate-500" />
          {(
            [
              ['latency', 'LATENCY'],
              ['score', 'SCORE'],
              ['vram', 'VRAM'],
            ] as const
          ).map(([key, label]) => (
            <button
              key={key}
              type="button"
              onClick={() => setSortBy(key)}
              className={`rounded px-2 py-0.5 transition ${
                sortBy === key
                  ? 'bg-amber-500/25 font-semibold text-amber-200'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      <div className="relative flex-1 space-y-2 overflow-y-auto pr-1">
        {sorted.map((entry) => {
          const meanPct = Math.max(4, Math.min(100, (entry.latency.mean_ms / maxMs) * 100))
          const p95Pct = Math.max(meanPct, Math.min(100, (entry.latency.p95_ms / maxMs) * 100))
          const isSelected = selectedIds.includes(entry.id)

          return (
            <button
              key={entry.id}
              type="button"
              onClick={() => onToggleSelect(entry.id)}
              className={`group block w-full rounded-lg border p-2 text-left transition ${
                isSelected
                  ? 'border-cyan-400/60 bg-cyan-500/10'
                  : 'border-slate-800 bg-slate-950/75 hover:border-slate-600'
              }`}
            >
              <div className="mb-1 flex items-center justify-between gap-2 font-mono text-[11px]">
                <div className="flex min-w-0 items-center gap-1.5">
                  <span className="truncate font-semibold text-slate-100">{entry.model}</span>
                  <span
                    className={`rounded border px-1.5 py-0 text-[9px] font-bold uppercase ${PRECISION_BADGE_CLASS[entry.precision]}`}
                  >
                    {entry.precision}
                  </span>
                  <span className="hidden text-[10px] text-slate-500 sm:inline">
                    {entry.denoise_steps}st · {entry.vram_gb.toFixed(1)}GB
                  </span>
                </div>
                <div className="flex shrink-0 items-center gap-2.5 text-[10px]">
                  <span className="font-bold text-white">
                    {entry.latency.mean_ms.toFixed(1)} ms
                  </span>
                  <span className="text-slate-400">p95 {entry.latency.p95_ms.toFixed(1)}</span>
                  <span className="w-14 text-right font-semibold text-emerald-300">
                    {entry.latency.fps.toFixed(1)} Hz
                  </span>
                </div>
              </div>

              <div className="relative h-2.5 w-full overflow-hidden rounded-full bg-slate-900">
                {/* 50ms / 20Hz budget marker */}
                <div
                  className="pointer-events-none absolute inset-y-0 z-10 w-px border-l border-dashed border-rose-400/70"
                  style={{ left: `${budget20HzPct}%` }}
                  title="50ms (20Hz real-time budget)"
                />
                {/* Mean bar */}
                <div
                  className={`h-full rounded-full bg-gradient-to-r ${PRECISION_BAR_CLASS[entry.precision]}`}
                  style={{ width: `${meanPct}%` }}
                />
                {/* P95 tick */}
                <div
                  className="pointer-events-none absolute top-0 h-full w-1 rounded-full bg-white/80"
                  style={{ left: `calc(${p95Pct}% - 2px)` }}
                  title={`p95: ${entry.latency.p95_ms.toFixed(1)}ms`}
                />
              </div>
            </button>
          )
        })}
      </div>
    </div>
  )
}
