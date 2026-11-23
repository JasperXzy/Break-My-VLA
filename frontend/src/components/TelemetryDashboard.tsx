import { useState } from 'react'
import { Activity, Flame, Gauge, ShieldCheck, TrendingUp } from 'lucide-react'
import {
  type ActionChunk,
  CHAOS_SKILL_META,
  type ChaosSkill,
  type FrameStatus,
} from '../types/protocol'
import type { LatencySample } from '../hooks/useVlaSession'

interface TelemetryDashboardProps {
  actionChunk: ActionChunk
  inferMs: number
  latencyHistory: LatencySample[]
  taskProgress: number
  activeChaos: ChaosSkill[]
  status: FrameStatus
}

type ChannelGroup = 'xyz' | 'rot'

const XYZ_CHANNELS = [
  { index: 0, label: 'ΔX', color: '#22d3ee' },
  { index: 1, label: 'ΔY', color: '#e879f9' },
  { index: 2, label: 'ΔZ', color: '#34d399' },
]

const ROT_CHANNELS = [
  { index: 3, label: 'RX', color: '#fbbf24' },
  { index: 4, label: 'RY', color: '#fb7185' },
  { index: 6, label: 'GRIP', color: '#60a5fa' },
]

function buildPolyline(
  chunk: ActionChunk,
  channelIndex: number,
  width: number,
  height: number,
  minVal: number,
  maxVal: number,
): string {
  if (chunk.length === 0) return ''
  const padX = 8
  const padY = 10
  const usableW = width - padX * 2
  const usableH = height - padY * 2
  const denom = Math.max(1, chunk.length - 1)

  return chunk
    .map((vec, i) => {
      const raw = vec[channelIndex] ?? 0
      const clamped = Math.max(minVal, Math.min(maxVal, raw))
      const norm = (clamped - minVal) / (maxVal - minVal)
      const x = padX + (i / denom) * usableW
      const y = padY + (1 - norm) * usableH
      return `${x.toFixed(1)},${y.toFixed(1)}`
    })
    .join(' ')
}

export function TelemetryDashboard({
  actionChunk,
  inferMs,
  latencyHistory,
  taskProgress,
  activeChaos,
  status,
}: TelemetryDashboardProps) {
  const [channelGroup, setChannelGroup] = useState<ChannelGroup>('xyz')
  const channels = channelGroup === 'xyz' ? XYZ_CHANNELS : ROT_CHANNELS
  const rangeMin = -1.05
  const rangeMax = 1.05

  const svgW = 360
  const svgH = 112
  const firstAction = actionChunk[0] ?? [0, 0, 0, 0, 0, 0, 0]
  const progressPct = Math.round(Math.max(0, Math.min(1, taskProgress)) * 100)

  const maxLatencyScale = Math.max(
    180,
    ...latencyHistory.map((s) => s.inferMs),
  )

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
      {/* Action Chunk Trajectory Chart */}
      <div className="arcade-panel rounded-xl p-3 lg:col-span-6">
        <div className="mb-2 flex items-center justify-between gap-2">
          <div className="flex items-center gap-1.5 font-mono text-[11px] font-semibold tracking-wider text-cyan-300">
            <TrendingUp className="h-3.5 w-3.5" />
            <span>ACTION CHUNK HORIZON ({actionChunk.length} STEPS)</span>
          </div>
          <div className="flex items-center gap-1 rounded-md border border-slate-700/80 bg-slate-950/80 p-0.5 font-mono text-[10px]">
            <button
              type="button"
              onClick={() => setChannelGroup('xyz')}
              className={`rounded px-2 py-0.5 transition ${
                channelGroup === 'xyz'
                  ? 'bg-cyan-500/25 font-semibold text-cyan-200'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              XYZ
            </button>
            <button
              type="button"
              onClick={() => setChannelGroup('rot')}
              className={`rounded px-2 py-0.5 transition ${
                channelGroup === 'rot'
                  ? 'bg-fuchsia-500/25 font-semibold text-fuchsia-200'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              ROT/GRIP
            </button>
          </div>
        </div>

        <div className="relative overflow-hidden rounded-lg border border-slate-800 bg-slate-950/90">
          <svg
            viewBox={`0 0 ${svgW} ${svgH}`}
            className="h-[112px] w-full"
            preserveAspectRatio="none"
          >
            {/* Horizontal gridlines */}
            {[0.2, 0.5, 0.8].map((ratio) => (
              <line
                key={ratio}
                x1={0}
                y1={svgH * ratio}
                x2={svgW}
                y2={svgH * ratio}
                stroke={ratio === 0.5 ? 'rgba(148,163,184,0.28)' : 'rgba(51,65,85,0.45)'}
                strokeDasharray={ratio === 0.5 ? '4 4' : '2 4'}
                strokeWidth={1}
              />
            ))}

            {/* Vertical horizon ticks */}
            {[0.25, 0.5, 0.75].map((ratio) => (
              <line
                key={ratio}
                x1={svgW * ratio}
                y1={0}
                x2={svgW * ratio}
                y2={svgH}
                stroke="rgba(51,65,85,0.35)"
                strokeWidth={1}
              />
            ))}

            {/* Action trajectories */}
            {channels.map((ch) => {
              const pts = buildPolyline(actionChunk, ch.index, svgW, svgH, rangeMin, rangeMax)
              if (!pts) return null
              return (
                <g key={ch.label}>
                  <polyline
                    points={pts}
                    fill="none"
                    stroke={ch.color}
                    strokeWidth={4}
                    strokeOpacity={0.18}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                  <polyline
                    points={pts}
                    fill="none"
                    stroke={ch.color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                </g>
              )
            })}
          </svg>
        </div>

        {/* Channel legend + t=0 vector values */}
        <div className="mt-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[10px]">
          <div className="flex items-center gap-3">
            {channels.map((ch) => (
              <span key={ch.label} className="flex items-center gap-1.5 text-slate-300">
                <span
                  className="inline-block h-2 w-2 rounded-full"
                  style={{ backgroundColor: ch.color, boxShadow: `0 0 8px ${ch.color}` }}
                />
                <span>{ch.label}:</span>
                <span className="font-semibold text-white">
                  {(firstAction[ch.index] ?? 0).toFixed(3)}
                </span>
              </span>
            ))}
          </div>
          <span className="text-slate-500">t+0 .. t+{Math.max(0, actionChunk.length - 1)}</span>
        </div>
      </div>

      {/* Inference Latency Bars + Task Progress + Active Chaos Badges */}
      <div className="arcade-panel flex flex-col justify-between gap-3 rounded-xl p-3 lg:col-span-6">
        {/* Latency spark-bar chart */}
        <div>
          <div className="mb-1.5 flex items-center justify-between font-mono text-[11px]">
            <div className="flex items-center gap-1.5 font-semibold tracking-wider text-amber-300">
              <Gauge className="h-3.5 w-3.5" />
              <span>INFERENCE LATENCY</span>
            </div>
            <div className="flex items-center gap-2">
              <span
                className={`font-bold ${
                  inferMs > 120
                    ? 'text-rose-400'
                    : inferMs > 55
                      ? 'text-amber-300'
                      : 'text-emerald-300'
                }`}
              >
                {inferMs.toFixed(1)} ms
              </span>
              <span className="text-[10px] text-slate-500">
                / {Math.round(maxLatencyScale)}ms scale
              </span>
            </div>
          </div>

          <div className="flex h-12 items-end gap-[3px] rounded-lg border border-slate-800 bg-slate-950/90 px-2 py-1.5">
            {Array.from({ length: 32 }).map((_, idx) => {
              const offset = 32 - latencyHistory.length
              const sample = idx >= offset ? latencyHistory[idx - offset] : null
              const val = sample?.inferMs ?? 0
              const pct = sample ? Math.max(8, Math.min(100, (val / maxLatencyScale) * 100)) : 4
              const barColor = !sample
                ? 'bg-slate-800/70'
                : val > 120
                  ? 'bg-rose-500 shadow-[0_0_8px_rgba(244,63,94,0.6)]'
                  : val > 55
                    ? 'bg-amber-400 shadow-[0_0_8px_rgba(251,191,36,0.5)]'
                    : 'bg-emerald-400 shadow-[0_0_8px_rgba(52,211,153,0.45)]'
              return (
                <div
                  key={idx}
                  className={`flex-1 rounded-t-sm transition-all duration-150 ${barColor}`}
                  style={{ height: `${pct}%` }}
                  title={sample ? `Step ${sample.step}: ${val.toFixed(1)} ms` : undefined}
                />
              )
            })}
          </div>
        </div>

        {/* Task Progress Bar */}
        <div>
          <div className="mb-1 flex items-center justify-between font-mono text-[11px]">
            <div className="flex items-center gap-1.5 font-semibold tracking-wider text-emerald-300">
              <Activity className="h-3.5 w-3.5" />
              <span>TASK COMPLETION</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="rounded border border-slate-700 bg-slate-900 px-1.5 py-0.5 text-[10px] uppercase text-slate-300">
                {status}
              </span>
              <span className="font-bold text-white">{progressPct}%</span>
            </div>
          </div>
          <div className="relative h-3.5 w-full overflow-hidden rounded-full border border-slate-700/80 bg-slate-950">
            <div
              className={`relative h-full rounded-full transition-all duration-200 ${
                status === 'success'
                  ? 'bg-gradient-to-r from-emerald-500 via-teal-400 to-cyan-300 shadow-[0_0_16px_rgba(16,185,129,0.75)]'
                  : 'bg-gradient-to-r from-cyan-500 via-sky-400 to-emerald-400 shadow-[0_0_14px_rgba(34,211,238,0.55)]'
              }`}
              style={{ width: `${progressPct}%` }}
            >
              <div className="animate-sheen absolute inset-y-0 w-16 bg-gradient-to-r from-transparent via-white/40 to-transparent" />
            </div>
          </div>
        </div>

        {/* Active Chaos Status Badges */}
        <div className="flex flex-wrap items-center gap-1.5 border-t border-slate-800/90 pt-2">
          <span className="mr-1 font-mono text-[10px] font-semibold uppercase tracking-wider text-slate-400">
            ACTIVE CHAOS:
          </span>
          {activeChaos.length === 0 ? (
            <span className="inline-flex items-center gap-1 rounded-md border border-emerald-500/35 bg-emerald-500/10 px-2 py-0.5 font-mono text-[10px] font-medium text-emerald-300">
              <ShieldCheck className="h-3 w-3" />
              NOMINAL // CLEAN FEED
            </span>
          ) : (
            activeChaos.map((skill) => {
              const meta = CHAOS_SKILL_META[skill]
              return (
                <span
                  key={skill}
                  className="inline-flex items-center gap-1 rounded-md border border-rose-400/60 bg-rose-500/20 px-2 py-0.5 font-mono text-[10px] font-bold text-rose-200 shadow-[0_0_10px_rgba(244,63,94,0.35)]"
                >
                  <Flame className="h-3 w-3 animate-pulse text-rose-400" />
                  {meta.labelZh}
                </span>
              )
            })
          )}
        </div>
      </div>
    </div>
  )
}
