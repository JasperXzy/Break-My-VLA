import { useMemo, useState } from 'react'
import { Crosshair, Sparkles } from 'lucide-react'
import {
  type BenchmarkEntry,
  formatBackendLabel,
  type InferenceBackend,
} from '../types/benchmark'
import type { Precision } from '../types/protocol'
import { formatAlignmentCosine, formatAlignmentError } from '../utils/arenaMarkdown'
import { computeParetoFrontier } from '../utils/pareto'

interface ParetoFrontierChartProps {
  entries: BenchmarkEntry[]
  contenderAId: string
  contenderBId: string
  onAddToCompare: (id: string) => void
}

const PRECISION_COLOR: Record<
  Precision,
  { fill: string; stroke: string; badge: string; label: string }
> = {
  bf16: {
    fill: '#34d399',
    stroke: '#a7f3d0',
    badge: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-200',
    label: 'BF16',
  },
  fp8: {
    fill: '#22d3ee',
    stroke: '#a5f3fc',
    badge: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200',
    label: 'FP8',
  },
  int8: {
    fill: '#fbbf24',
    stroke: '#fde68a',
    badge: 'border-amber-400/50 bg-amber-500/15 text-amber-200',
    label: 'INT8',
  },
  nvfp4: {
    fill: '#e879f9',
    stroke: '#f5d0fe',
    badge: 'border-fuchsia-400/50 bg-fuchsia-500/15 text-fuchsia-200',
    label: 'NVFP4',
  },
}

function buildLatencyTicks(maxMs: number): number[] {
  const step = maxMs <= 80 ? 10 : maxMs <= 160 ? 20 : maxMs <= 300 ? 50 : 100
  const ticks: number[] = []
  for (let v = 0; v <= maxMs; v += step) {
    ticks.push(v)
  }
  if (ticks[ticks.length - 1] < maxMs) {
    ticks.push(maxMs)
  }
  return ticks
}

function diamondPoints(cx: number, cy: number, r: number): string {
  return `${cx.toFixed(1)},${(cy - r).toFixed(1)} ${(cx + r).toFixed(1)},${cy.toFixed(1)} ${cx.toFixed(1)},${(cy + r).toFixed(1)} ${(cx - r).toFixed(1)},${cy.toFixed(1)}`
}

export function ParetoFrontierChart({
  entries,
  contenderAId,
  contenderBId,
  onAddToCompare,
}: ParetoFrontierChartProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const frontierEntries = useMemo(() => computeParetoFrontier(entries), [entries])
  const frontierIds = useMemo(() => new Set(frontierEntries.map((e) => e.id)), [frontierEntries])

  const hoveredEntry = useMemo(
    () => (hoveredId ? entries.find((e) => e.id === hoveredId) ?? null : null),
    [entries, hoveredId],
  )

  if (entries.length === 0) {
    return null
  }

  // SVG plot geometry
  const width = 720
  const height = 340
  const padLeft = 52
  const padRight = 24
  const padTop = 28
  const padBottom = 46
  const plotW = width - padLeft - padRight
  const plotH = height - padTop - padBottom

  const rawMaxMs = Math.max(...entries.map((e) => e.latency.mean_ms), 55)
  const maxMs = Math.max(60, Math.ceil((rawMaxMs * 1.15) / 10) * 10)
  const xTicks = buildLatencyTicks(maxMs)
  const yTicks = [0, 20, 40, 60, 80, 100]

  const toX = (ms: number) => padLeft + (Math.max(0, Math.min(maxMs, ms)) / maxMs) * plotW
  const toY = (score: number) =>
    padTop + plotH - (Math.max(0, Math.min(100, score)) / 100) * plotH

  const x33 = toX(33)
  const x50 = toX(50)

  const frontierPolyline = frontierEntries
    .map((e) => `${toX(e.latency.mean_ms).toFixed(1)},${toY(e.overall_score).toFixed(1)}`)
    .join(' ')

  return (
    <section
      aria-label="Quantization Pareto Frontier"
      className="arcade-panel rounded-xl p-3.5"
    >
      {/* Header + Legend */}
      <div className="mb-3 flex flex-wrap items-start justify-between gap-2.5 border-b border-slate-800/90 pb-2.5">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2 font-mono text-xs font-bold tracking-wider text-cyan-300">
            <Crosshair className="h-4 w-4 shrink-0 text-emerald-300" />
            <span>QUANTIZATION PARETO FRONTIER // 延迟 vs 抗压得分帕累托前沿</span>
            <span
              data-testid="pareto-count-badge"
              className="inline-flex items-center gap-1 rounded border border-emerald-400/50 bg-emerald-500/15 px-1.5 py-0.5 text-[10px] font-bold text-emerald-200"
            >
              <Sparkles className="h-3 w-3" />
              {frontierEntries.length} PARETO OPTIMAL
            </span>
          </div>
          <p className="mt-0.5 font-mono text-[10px] text-slate-400">
            越靠左上角越优（更低延迟 + 更高抗压）· 悬停查看精度对齐详情，点击散点直接加入 1v1 对比
          </p>
        </div>

        {/* Precision Color + Backend Shape Legend */}
        <div className="flex flex-wrap items-center gap-2 font-mono text-[10px]">
          <div className="flex flex-wrap items-center gap-1.5 rounded-lg border border-slate-800 bg-slate-950/80 px-2 py-1">
            {(['bf16', 'fp8', 'int8', 'nvfp4'] as const).map((prec) => (
              <span key={prec} className="inline-flex items-center gap-1 text-slate-300">
                <span
                  className="h-2 w-2 rounded-full"
                  style={{ backgroundColor: PRECISION_COLOR[prec].fill }}
                />
                <span>{PRECISION_COLOR[prec].label}</span>
              </span>
            ))}
          </div>
          <div className="flex items-center gap-2 rounded-lg border border-slate-800 bg-slate-950/80 px-2 py-1 text-slate-300">
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2.5 w-2.5 rounded-full border border-slate-200 bg-slate-400/70" />
              <span>PyTorch</span>
            </span>
            <span className="inline-flex items-center gap-1">
              <span className="inline-block h-2 w-2 rotate-45 border border-cyan-200 bg-cyan-400/80" />
              <span>TensorRT</span>
            </span>
          </div>
        </div>
      </div>

      {/* Main Plot + Side Inspector / Hover Card */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-12">
        <div className="relative overflow-x-auto lg:col-span-9">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            className="h-auto min-h-[250px] w-full min-w-[320px] select-none"
            role="img"
            aria-label="Quantization Pareto Frontier scatter plot of latency versus resilience score"
          >
            <defs>
              <linearGradient id="paretoRtZone" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="rgba(16, 185, 129, 0.12)" />
                <stop offset="100%" stopColor="rgba(34, 211, 238, 0.04)" />
              </linearGradient>
              <filter id="paretoGlow" x="-20%" y="-20%" width="140%" height="140%">
                <feGaussianBlur stdDeviation="2.5" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Real-time <= 33ms zone shading */}
            <rect
              x={padLeft}
              y={padTop}
              width={Math.max(0, x33 - padLeft)}
              height={plotH}
              fill="url(#paretoRtZone)"
            />

            {/* Plot frame background */}
            <rect
              x={padLeft}
              y={padTop}
              width={plotW}
              height={plotH}
              fill="rgba(2, 6, 23, 0.45)"
              stroke="rgba(51, 65, 85, 0.7)"
              strokeWidth={1}
              rx={4}
            />

            {/* Horizontal Y gridlines & labels */}
            {yTicks.map((tick) => {
              const y = toY(tick)
              return (
                <g key={`y-${tick}`}>
                  <line
                    x1={padLeft}
                    y1={y}
                    x2={padLeft + plotW}
                    y2={y}
                    stroke="rgba(51, 65, 85, 0.45)"
                    strokeDasharray={tick === 0 ? undefined : '3 4'}
                    strokeWidth={1}
                  />
                  <text
                    x={padLeft - 8}
                    y={y + 3}
                    textAnchor="end"
                    fill="#94a3b8"
                    fontSize="9.5"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {tick}
                  </text>
                </g>
              )
            })}

            {/* Vertical X gridlines & labels */}
            {xTicks.map((tick) => {
              const x = toX(tick)
              return (
                <g key={`x-${tick}`}>
                  <line
                    x1={x}
                    y1={padTop}
                    x2={x}
                    y2={padTop + plotH}
                    stroke="rgba(51, 65, 85, 0.35)"
                    strokeDasharray={tick === 0 ? undefined : '3 4'}
                    strokeWidth={1}
                  />
                  <text
                    x={x}
                    y={padTop + plotH + 16}
                    textAnchor="middle"
                    fill="#94a3b8"
                    fontSize="9.5"
                    fontFamily="JetBrains Mono, monospace"
                  >
                    {tick}ms
                  </text>
                </g>
              )
            })}

            {/* Axis Titles */}
            <text
              x={padLeft + plotW / 2}
              y={height - 8}
              textAnchor="middle"
              fill="#cbd5e1"
              fontSize="10"
              fontWeight="600"
              fontFamily="JetBrains Mono, monospace"
            >
              INFERENCE LATENCY (ms · lower is faster)
            </text>
            <text
              x={14}
              y={padTop + plotH / 2}
              textAnchor="middle"
              fill="#cbd5e1"
              fontSize="10"
              fontWeight="600"
              fontFamily="JetBrains Mono, monospace"
              transform={`rotate(-90 14 ${padTop + plotH / 2})`}
            >
              RESILIENCE SCORE (0–100)
            </text>

            {/* Real-Time Control Redline 1: 33ms / 30Hz */}
            <g data-testid="rt-line-33ms">
              <line
                x1={x33}
                y1={padTop}
                x2={x33}
                y2={padTop + plotH}
                stroke="#f43f5e"
                strokeWidth={1.6}
                strokeDasharray="5 3"
              />
              <rect
                x={Math.max(padLeft + 2, x33 - 42)}
                y={padTop - 20}
                width={84}
                height={16}
                rx={3}
                fill="rgba(136, 19, 55, 0.85)"
                stroke="rgba(244, 63, 94, 0.8)"
                strokeWidth={0.8}
              />
              <text
                x={Math.max(padLeft + 44, x33)}
                y={padTop - 9}
                textAnchor="middle"
                fill="#fecdd3"
                fontSize="8.5"
                fontWeight="700"
                fontFamily="JetBrains Mono, monospace"
              >
                33ms / 30Hz
              </text>
            </g>

            {/* Real-Time Control Redline 2: 50ms / 20Hz */}
            <g data-testid="rt-line-50ms">
              <line
                x1={x50}
                y1={padTop}
                x2={x50}
                y2={padTop + plotH}
                stroke="#fb7185"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                opacity={0.9}
              />
              <rect
                x={Math.min(width - padRight - 84, x50 + 4)}
                y={padTop - 20}
                width={84}
                height={16}
                rx={3}
                fill="rgba(120, 53, 15, 0.85)"
                stroke="rgba(251, 113, 133, 0.75)"
                strokeWidth={0.8}
              />
              <text
                x={Math.min(width - padRight - 42, x50 + 46)}
                y={padTop - 9}
                textAnchor="middle"
                fill="#fed7aa"
                fontSize="8.5"
                fontWeight="700"
                fontFamily="JetBrains Mono, monospace"
              >
                50ms / 20Hz
              </text>
            </g>

            {/* Pareto Optimal Frontier Curve */}
            {frontierEntries.length >= 2 && (
              <g data-testid="pareto-frontier-curve">
                <polyline
                  points={frontierPolyline}
                  fill="none"
                  stroke="rgba(52, 211, 153, 0.35)"
                  strokeWidth={6}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  filter="url(#paretoGlow)"
                />
                <polyline
                  points={frontierPolyline}
                  fill="none"
                  stroke="#34d399"
                  strokeWidth={2.2}
                  strokeDasharray="6 3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
              </g>
            )}

            {/* Scatter Points */}
            {entries.map((entry) => {
              const cx = toX(entry.latency.mean_ms)
              const cy = toY(entry.overall_score)
              const isPareto = frontierIds.has(entry.id)
              const isHovered = hoveredId === entry.id
              const isA = entry.id === contenderAId
              const isB = entry.id === contenderBId
              const backend: InferenceBackend = entry.backend ?? 'pytorch'
              const isTrt = backend === 'tensorrt'
              const color = PRECISION_COLOR[entry.precision]
              const radius = isHovered ? 9 : isPareto ? 7.2 : 5.8

              return (
                <g
                  key={entry.id}
                  data-testid={`pareto-point-${entry.id}`}
                  data-pareto={isPareto ? 'true' : 'false'}
                  data-backend={backend}
                  role="button"
                  tabIndex={0}
                  aria-label={`${entry.model} ${entry.precision.toUpperCase()} ${formatBackendLabel(backend)} latency ${entry.latency.mean_ms}ms score ${entry.overall_score}`}
                  className="cursor-pointer outline-none"
                  onMouseEnter={() => setHoveredId(entry.id)}
                  onMouseLeave={() => setHoveredId((prev) => (prev === entry.id ? null : prev))}
                  onFocus={() => setHoveredId(entry.id)}
                  onBlur={() => setHoveredId((prev) => (prev === entry.id ? null : prev))}
                  onClick={() => onAddToCompare(entry.id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onAddToCompare(entry.id)
                    }
                  }}
                >
                  {/* Pareto optimal halo */}
                  {isPareto && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={radius + 4}
                      fill="none"
                      stroke="rgba(52, 211, 153, 0.55)"
                      strokeWidth={1.3}
                    />
                  )}

                  {/* Active 1v1 Contender A / B highlight ring */}
                  {(isA || isB) && (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={radius + 6.5}
                      fill="none"
                      stroke={isA ? '#22d3ee' : '#e879f9'}
                      strokeWidth={1.6}
                      strokeDasharray="3 2"
                    />
                  )}

                  {/* Point shape: Diamond for TensorRT, Circle for PyTorch */}
                  {isTrt ? (
                    <polygon
                      points={diamondPoints(cx, cy, radius + 1.2)}
                      fill={color.fill}
                      stroke={isHovered ? '#ffffff' : color.stroke}
                      strokeWidth={isHovered ? 2.2 : 1.4}
                      filter={isPareto || isHovered ? 'url(#paretoGlow)' : undefined}
                    />
                  ) : (
                    <circle
                      cx={cx}
                      cy={cy}
                      r={radius}
                      fill={color.fill}
                      stroke={isHovered ? '#ffffff' : color.stroke}
                      strokeWidth={isHovered ? 2.2 : 1.4}
                      filter={isPareto || isHovered ? 'url(#paretoGlow)' : undefined}
                    />
                  )}

                  {/* Contender A/B mini label */}
                  {(isA || isB) && (
                    <text
                      x={cx}
                      y={cy - radius - 8}
                      textAnchor="middle"
                      fill={isA ? '#67e8f9' : '#f0abfc'}
                      fontSize="9"
                      fontWeight="800"
                      fontFamily="JetBrains Mono, monospace"
                    >
                      {isA && isB ? 'A/B' : isA ? 'A' : 'B'}
                    </text>
                  )}
                </g>
              )
            })}
          </svg>
        </div>

        {/* Right Detail / Hover Tooltip Card + Frontier Summary */}
        <div className="flex flex-col justify-between gap-2.5 lg:col-span-3">
          {hoveredEntry ? (
            <div
              data-testid="pareto-tooltip"
              className="rounded-xl border border-cyan-400/50 bg-slate-950/95 p-3 font-mono text-xs shadow-[0_0_24px_rgba(34,211,238,0.16)]"
            >
              <div className="flex flex-wrap items-center justify-between gap-1">
                <span className="font-bold text-white">{hoveredEntry.model}</span>
                {frontierIds.has(hoveredEntry.id) && (
                  <span className="rounded border border-emerald-400/50 bg-emerald-500/20 px-1.5 py-0 text-[9px] font-bold text-emerald-200">
                    PARETO
                  </span>
                )}
              </div>

              <div className="mt-1.5 flex flex-wrap items-center gap-1.5">
                <span
                  className={`rounded border px-1.5 py-0.5 text-[9px] font-bold uppercase ${PRECISION_COLOR[hoveredEntry.precision].badge}`}
                >
                  {hoveredEntry.precision} · {hoveredEntry.denoise_steps}st
                </span>
                <span
                  className={`rounded border px-1.5 py-0.5 text-[9px] font-bold ${
                    hoveredEntry.backend === 'tensorrt'
                      ? 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200'
                      : 'border-slate-600 bg-slate-800/80 text-slate-300'
                  }`}
                >
                  {hoveredEntry.backend === 'tensorrt' ? '◆ TensorRT' : '● PyTorch'}
                </span>
              </div>

              <div className="mt-2.5 space-y-1 text-[11px] text-slate-300">
                <div className="flex justify-between">
                  <span className="text-slate-400">Resilience</span>
                  <strong className="text-emerald-300">
                    {hoveredEntry.overall_score.toFixed(1)} / 100
                  </strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">Mean / FPS</span>
                  <strong className="text-amber-300">
                    {hoveredEntry.latency.mean_ms.toFixed(1)}ms · {hoveredEntry.latency.fps.toFixed(1)}Hz
                  </strong>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">P95 / P99</span>
                  <span className="text-slate-200">
                    {hoveredEntry.latency.p95_ms.toFixed(1)}
                    {hoveredEntry.latency.p99_ms !== undefined
                      ? ` / ${hoveredEntry.latency.p99_ms.toFixed(1)}ms`
                      : 'ms'}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-slate-400">VRAM</span>
                  <span className="text-slate-200">{hoveredEntry.vram_gb.toFixed(2)} GB</span>
                </div>
                {hoveredEntry.alignment && (
                  <>
                    <div className="my-1 border-t border-slate-800" />
                    <div className="flex justify-between">
                      <span className="text-slate-400">Cos vs BF16</span>
                      <strong className="text-cyan-300">
                        {formatAlignmentCosine(hoveredEntry.alignment.action_cos_vs_bf16)}
                      </strong>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-slate-400">MSE vs BF16</span>
                      <span className="text-fuchsia-300">
                        {formatAlignmentError(hoveredEntry.alignment.action_mse_vs_bf16)}
                      </span>
                    </div>
                  </>
                )}
              </div>

              <div className="mt-2.5 rounded border border-cyan-500/30 bg-cyan-500/10 px-2 py-1 text-center text-[10px] text-cyan-200">
                Click point to pin into 1v1 Arena
              </div>
            </div>
          ) : (
            <div className="rounded-xl border border-slate-800 bg-slate-950/80 p-3 font-mono text-[11px] text-slate-400">
              <div className="font-bold text-slate-200">FRONTIER INSPECTOR</div>
              <p className="mt-1 text-[10px] leading-relaxed">
                Hover any scatter point to inspect P99 latency & BF16 action-chunk drift. Click any
                point to pair it directly into the 1v1 Head-to-Head Arena above.
              </p>
            </div>
          )}

          {/* Pareto Frontier Quick-Pick List */}
          <div className="rounded-xl border border-slate-800 bg-slate-950/75 p-2.5 font-mono">
            <div className="mb-1.5 flex items-center justify-between text-[10px] font-bold text-emerald-300">
              <span>PARETO FRONTIER ({frontierEntries.length})</span>
              <span className="text-slate-500">CLICK TO 1V1</span>
            </div>
            <div className="space-y-1">
              {frontierEntries.map((entry) => {
                const isSelected = entry.id === contenderAId || entry.id === contenderBId
                return (
                  <button
                    key={entry.id}
                    type="button"
                    onMouseEnter={() => setHoveredId(entry.id)}
                    onMouseLeave={() => setHoveredId((prev) => (prev === entry.id ? null : prev))}
                    onClick={() => onAddToCompare(entry.id)}
                    className={`flex w-full items-center justify-between gap-1.5 rounded-lg border px-2 py-1 text-left text-[10px] transition ${
                      isSelected
                        ? 'border-cyan-400/60 bg-cyan-500/15 text-white'
                        : 'border-slate-800/90 bg-slate-900/70 text-slate-300 hover:border-emerald-400/50 hover:text-white'
                    }`}
                  >
                    <span className="min-w-0 truncate font-semibold">
                      {entry.backend === 'tensorrt' ? '◆ ' : '● '}
                      {entry.model}{' '}
                      <span className="uppercase text-slate-400">{entry.precision}</span>
                    </span>
                    <span className="shrink-0 text-emerald-300">
                      {entry.latency.mean_ms.toFixed(1)}ms · {entry.overall_score.toFixed(1)}
                    </span>
                  </button>
                )
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  )
}
