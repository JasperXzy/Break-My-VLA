import { useState } from 'react'
import { ShieldAlert } from 'lucide-react'
import { type BenchmarkEntry, RADAR_AXES } from '../types/benchmark'

interface StressRadarChartProps {
  entries: BenchmarkEntry[]
}

const SERIES_COLORS = [
  { stroke: '#22d3ee', fill: 'rgba(34, 211, 238, 0.20)', dot: '#67e8f9' },
  { stroke: '#e879f9', fill: 'rgba(232, 121, 249, 0.18)', dot: '#f0abfc' },
  { stroke: '#34d399', fill: 'rgba(52, 211, 153, 0.18)', dot: '#6ee7b7' },
  { stroke: '#fbbf24', fill: 'rgba(251, 191, 36, 0.18)', dot: '#fde68a' },
]

function polarToCartesian(cx: number, cy: number, radius: number, angleIndex: number, total: number) {
  const angle = -Math.PI / 2 + (angleIndex * 2 * Math.PI) / total
  return {
    x: cx + radius * Math.cos(angle),
    y: cy + radius * Math.sin(angle),
  }
}

export function StressRadarChart({ entries }: StressRadarChartProps) {
  const [hoveredId, setHoveredId] = useState<string | null>(null)

  const size = 310
  const cx = size / 2
  const cy = size / 2 + 4
  const maxR = 104
  const rings = [20, 40, 60, 80, 100]
  const axisCount = RADAR_AXES.length

  const visible = entries.slice(0, 4)

  return (
    <div className="arcade-panel flex h-full flex-col rounded-xl p-3.5">
      <div className="mb-2 flex items-center justify-between gap-2">
        <div>
          <div className="flex items-center gap-1.5 font-mono text-xs font-bold tracking-wider text-cyan-300">
            <ShieldAlert className="h-4 w-4" />
            <span>5-DIM STRESS RESILIENCE RADAR</span>
          </div>
          <p className="font-mono text-[10px] text-slate-400">
            五维抗压雷达图 · 点击下方榜单可切换最多 4 组对比
          </p>
        </div>
        <span className="rounded border border-slate-700 bg-slate-900/90 px-2 py-0.5 font-mono text-[10px] text-slate-300">
          {visible.length} ACTIVE
        </span>
      </div>

      <div className="relative flex flex-1 items-center justify-center">
        <svg
          viewBox={`0 0 ${size} ${size}`}
          className="h-[275px] w-full max-w-[330px]"
          role="img"
          aria-label="5-dimensional stress resilience radar chart"
        >
          {/* Concentric pentagon grid rings */}
          {rings.map((ring) => {
            const r = (ring / 100) * maxR
            const pts = RADAR_AXES.map((_, i) => {
              const p = polarToCartesian(cx, cy, r, i, axisCount)
              return `${p.x.toFixed(1)},${p.y.toFixed(1)}`
            }).join(' ')
            return (
              <g key={ring}>
                <polygon
                  points={pts}
                  fill="none"
                  stroke={ring === 100 ? 'rgba(148,163,184,0.35)' : 'rgba(51,65,85,0.55)'}
                  strokeDasharray={ring === 100 ? undefined : '3 3'}
                  strokeWidth={ring === 100 ? 1.2 : 1}
                />
                <text
                  x={cx + 4}
                  y={cy - r + 10}
                  fill="#64748b"
                  fontSize="8.5"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {ring}
                </text>
              </g>
            )
          })}

          {/* Radial axis spokes & labels */}
          {RADAR_AXES.map((axis, idx) => {
            const end = polarToCartesian(cx, cy, maxR, idx, axisCount)
            const labelPos = polarToCartesian(cx, cy, maxR + 24, idx, axisCount)
            return (
              <g key={axis.key}>
                <line
                  x1={cx}
                  y1={cy}
                  x2={end.x}
                  y2={end.y}
                  stroke="rgba(71,85,105,0.65)"
                  strokeWidth={1}
                />
                <text
                  x={labelPos.x}
                  y={labelPos.y - 2}
                  textAnchor="middle"
                  fill="#e2e8f0"
                  fontSize="10"
                  fontWeight="700"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {axis.labelZh}
                </text>
                <text
                  x={labelPos.x}
                  y={labelPos.y + 10}
                  textAnchor="middle"
                  fill="#64748b"
                  fontSize="8.5"
                  fontFamily="JetBrains Mono, monospace"
                >
                  {axis.short}
                </text>
              </g>
            )
          })}

          {/* Data polygons */}
          {visible.map((entry, seriesIdx) => {
            const palette = SERIES_COLORS[seriesIdx % SERIES_COLORS.length]
            const dimmed = hoveredId !== null && hoveredId !== entry.id
            const coords = RADAR_AXES.map((axis, i) => {
              const score = entry.radar_scores[axis.key]
              return polarToCartesian(cx, cy, (score / 100) * maxR, i, axisCount)
            })
            const poly = coords.map((p) => `${p.x.toFixed(1)},${p.y.toFixed(1)}`).join(' ')

            return (
              <g
                key={entry.id}
                opacity={dimmed ? 0.25 : 1}
                className="transition-opacity duration-150"
              >
                <polygon
                  points={poly}
                  fill={palette.fill}
                  stroke={palette.stroke}
                  strokeWidth={hoveredId === entry.id ? 2.6 : 2}
                  strokeLinejoin="round"
                />
                {coords.map((pt, i) => (
                  <circle
                    key={RADAR_AXES[i].key}
                    cx={pt.x}
                    cy={pt.y}
                    r={hoveredId === entry.id ? 4 : 3}
                    fill={palette.dot}
                    stroke="#060912"
                    strokeWidth={1.2}
                  />
                ))}
              </g>
            )
          })}
        </svg>
      </div>

      {/* Interactive Legend */}
      <div className="mt-2 grid grid-cols-1 gap-1.5 border-t border-slate-800/90 pt-2.5 sm:grid-cols-2">
        {visible.map((entry, idx) => {
          const palette = SERIES_COLORS[idx % SERIES_COLORS.length]
          return (
            <div
              key={entry.id}
              onMouseEnter={() => setHoveredId(entry.id)}
              onMouseLeave={() => setHoveredId(null)}
              className="flex cursor-default items-center justify-between rounded-lg border border-slate-800 bg-slate-950/75 px-2.5 py-1.5 font-mono text-[10px] transition hover:border-slate-600"
            >
              <div className="flex min-w-0 items-center gap-2">
                <span
                  className="h-2.5 w-2.5 shrink-0 rounded-full"
                  style={{
                    backgroundColor: palette.stroke,
                    boxShadow: `0 0 8px ${palette.stroke}`,
                  }}
                />
                <span className="truncate font-semibold text-slate-100">
                  {entry.model}
                </span>
                <span className="rounded bg-slate-800 px-1 py-0.5 text-[9px] uppercase text-slate-300">
                  {entry.precision}
                </span>
              </div>
              <span className="ml-2 font-bold text-white">{entry.overall_score.toFixed(1)}</span>
            </div>
          )
        })}
      </div>
    </div>
  )
}
