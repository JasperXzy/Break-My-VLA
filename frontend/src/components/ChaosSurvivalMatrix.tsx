import { useMemo } from 'react'
import { Flame, Grid3X3, ShieldAlert } from 'lucide-react'
import { type BenchmarkEntry, RADAR_AXES, type RadarScores } from '../types/benchmark'
import type { Precision } from '../types/protocol'

interface ChaosSurvivalMatrixProps {
  entries: BenchmarkEntry[]
  contenderAId: string
  contenderBId: string
  onPickContenderA: (id: string) => void
  onPickContenderB: (id: string) => void
}

const PRECISION_TAG: Record<Precision, string> = {
  bf16: 'text-emerald-300 border-emerald-400/50 bg-emerald-500/15',
  fp8: 'text-cyan-300 border-cyan-400/50 bg-cyan-500/15',
  int8: 'text-amber-300 border-amber-400/50 bg-amber-500/15',
  nvfp4: 'text-fuchsia-300 border-fuchsia-400/50 bg-fuchsia-500/15',
}

interface HeatCellStyle {
  cellClass: string
  tier: string
}

function getHeatmapStyle(score: number): HeatCellStyle {
  if (score >= 85) {
    return {
      cellClass: 'border-emerald-400/50 bg-emerald-500/25 text-emerald-100',
      tier: 'SURVIVED',
    }
  }
  if (score >= 70) {
    return {
      cellClass: 'border-teal-400/40 bg-teal-500/20 text-teal-100',
      tier: 'STABLE',
    }
  }
  if (score >= 55) {
    return {
      cellClass: 'border-amber-400/45 bg-amber-500/20 text-amber-100',
      tier: 'SHAKY',
    }
  }
  if (score >= 35) {
    return {
      cellClass: 'border-orange-400/55 bg-orange-500/25 text-orange-100',
      tier: 'FRAGILE',
    }
  }
  return {
    cellClass: 'border-rose-400/70 bg-rose-600/35 text-rose-100 shadow-[inset_0_0_12px_rgba(244,63,94,0.3)]',
    tier: 'BROKE',
  }
}

function findWeakestAxis(scores: RadarScores): { key: keyof RadarScores; labelEn: string; score: number } {
  let weakest = RADAR_AXES[0]
  let minScore = scores[weakest.key]
  for (const axis of RADAR_AXES) {
    const val = scores[axis.key]
    if (val < minScore) {
      minScore = val
      weakest = axis
    }
  }
  return { key: weakest.key, labelEn: weakest.labelEn, score: minScore }
}

export function ChaosSurvivalMatrix({
  entries,
  contenderAId,
  contenderBId,
  onPickContenderA,
  onPickContenderB,
}: ChaosSurvivalMatrixProps) {
  const columnAverages = useMemo(() => {
    if (entries.length === 0) return null
    const avgs = RADAR_AXES.map((axis) => {
      const sum = entries.reduce((acc, e) => acc + e.radar_scores[axis.key], 0)
      return {
        key: axis.key,
        labelEn: axis.labelEn,
        avg: Number((sum / entries.length).toFixed(1)),
      }
    })
    let mostLethal = avgs[0]
    for (const item of avgs) {
      if (item.avg < mostLethal.avg) mostLethal = item
    }
    return { avgs, mostLethal }
  }, [entries])

  if (entries.length === 0) {
    return null
  }

  return (
    <section
      aria-label="Chaos Survival Matrix"
      className="arcade-panel overflow-hidden rounded-xl"
    >
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-800 px-3.5 py-2.5">
        <div>
          <div className="flex items-center gap-1.5 font-mono text-xs font-bold tracking-wider text-emerald-300">
            <Grid3X3 className="h-4 w-4 text-rose-400" />
            <span>CHAOS SURVIVAL MATRIX // 整蛊存活热力矩阵</span>
          </div>
          <p className="font-mono text-[10px] text-slate-400">
            按模型×精度为行、五维整蛊为列展示红绿存活热力，一眼定位各模型最先破防项
          </p>
        </div>

        {/* Heat Legend */}
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-[9px]">
          <span className="rounded border border-rose-400/70 bg-rose-600/35 px-1.5 py-0.5 text-rose-100">
            0-34 BROKE
          </span>
          <span className="rounded border border-orange-400/55 bg-orange-500/25 px-1.5 py-0.5 text-orange-100">
            35-54 FRAGILE
          </span>
          <span className="rounded border border-amber-400/45 bg-amber-500/20 px-1.5 py-0.5 text-amber-100">
            55-69 SHAKY
          </span>
          <span className="rounded border border-teal-400/40 bg-teal-500/20 px-1.5 py-0.5 text-teal-100">
            70-84 STABLE
          </span>
          <span className="rounded border border-emerald-400/50 bg-emerald-500/25 px-1.5 py-0.5 text-emerald-100">
            85+ SURVIVED
          </span>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-left font-mono text-xs">
          <thead>
            <tr className="border-b border-slate-800 bg-slate-950/85 text-[10px] uppercase tracking-wider text-slate-400">
              <th className="px-3 py-2">MODEL × PRECISION</th>
              {RADAR_AXES.map((axis) => (
                <th key={axis.key} className="px-2 py-2 text-center">
                  <div className="text-slate-200">{axis.labelEn}</div>
                  <div className="text-[9px] font-normal text-slate-500">{axis.labelZh}</div>
                </th>
              ))}
              <th className="px-3 py-2 text-center">FIRST TO BREAK</th>
              <th className="px-3 py-2 text-right">1V1 PIN</th>
            </tr>
          </thead>

          <tbody className="divide-y divide-slate-800/70">
            {entries.map((entry) => {
              const weakest = findWeakestAxis(entry.radar_scores)
              const isA = entry.id === contenderAId
              const isB = entry.id === contenderBId

              return (
                <tr key={entry.id} className="hover:bg-slate-900/60">
                  <td className="px-3 py-2">
                    <div className="flex flex-wrap items-center gap-1.5">
                      <span className="font-semibold text-white">{entry.model}</span>
                      <span
                        className={`rounded border px-1.5 py-0 text-[9px] font-bold uppercase ${PRECISION_TAG[entry.precision]}`}
                      >
                        {entry.precision}
                      </span>
                      {entry.backend && (
                        <span
                          className={`rounded border px-1 py-0 text-[9px] font-bold ${
                            entry.backend === 'tensorrt'
                              ? 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200'
                              : 'border-slate-700 bg-slate-900 text-slate-400'
                          }`}
                        >
                          {entry.backend === 'tensorrt' ? '◆ TRT' : '● PT'}
                        </span>
                      )}
                      <span className="text-[10px] text-slate-500">{entry.denoise_steps}st</span>
                    </div>
                  </td>

                  {RADAR_AXES.map((axis) => {
                    const score = entry.radar_scores[axis.key]
                    const style = getHeatmapStyle(score)
                    const isWeakestCell = axis.key === weakest.key && score < 85
                    return (
                      <td key={axis.key} className="px-1.5 py-1.5 text-center">
                        <div
                          data-testid={`heatmap-cell-${entry.id}-${axis.key}`}
                          className={`mx-auto flex min-w-[74px] flex-col items-center justify-center rounded-lg border px-2 py-1.5 transition ${style.cellClass}`}
                          title={`${entry.model} (${entry.precision.toUpperCase()}) · ${axis.labelEn}: ${score}% (${style.tier})`}
                        >
                          <span className="text-xs font-extrabold">{score.toFixed(0)}%</span>
                          <span className="text-[8px] tracking-wider opacity-85">
                            {isWeakestCell ? '⚠ BREAK' : style.tier}
                          </span>
                        </div>
                      </td>
                    )
                  })}

                  <td className="px-3 py-2 text-center">
                    <span
                      className={`inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[10px] font-bold ${
                        weakest.score < 55
                          ? 'border-rose-400/60 bg-rose-500/20 text-rose-200'
                          : 'border-amber-400/45 bg-amber-500/15 text-amber-200'
                      }`}
                    >
                      <Flame className="h-3 w-3" />
                      {weakest.labelEn} ({weakest.score.toFixed(0)}%)
                    </span>
                  </td>

                  <td className="px-3 py-2 text-right">
                    <div className="inline-flex items-center gap-1">
                      <button
                        type="button"
                        onClick={() => onPickContenderA(entry.id)}
                        className={`rounded border px-1.5 py-0.5 text-[10px] font-bold transition ${
                          isA
                            ? 'border-cyan-400 bg-cyan-500/30 text-cyan-100'
                            : 'border-slate-700 bg-slate-900 text-slate-400 hover:border-cyan-400/60 hover:text-cyan-200'
                        }`}
                        title="Set as Contender A in 1v1 Card"
                      >
                        A
                      </button>
                      <button
                        type="button"
                        onClick={() => onPickContenderB(entry.id)}
                        className={`rounded border px-1.5 py-0.5 text-[10px] font-bold transition ${
                          isB
                            ? 'border-fuchsia-400 bg-fuchsia-500/30 text-fuchsia-100'
                            : 'border-slate-700 bg-slate-900 text-slate-400 hover:border-fuchsia-400/60 hover:text-fuchsia-200'
                        }`}
                        title="Set as Contender B in 1v1 Card"
                      >
                        B
                      </button>
                    </div>
                  </td>
                </tr>
              )
            })}
          </tbody>

          {columnAverages && (
            <tfoot>
              <tr className="border-t border-slate-700/90 bg-slate-950/90 text-[10px]">
                <td className="px-3 py-2 font-bold text-slate-300">
                  SUITE AXIS AVG ({entries.length} configs)
                </td>
                {columnAverages.avgs.map((col) => {
                  const isLethal = col.key === columnAverages.mostLethal.key
                  return (
                    <td key={col.key} className="px-2 py-2 text-center">
                      <span
                        className={`font-bold ${isLethal ? 'text-rose-300' : 'text-slate-200'}`}
                      >
                        {col.avg.toFixed(1)}%
                      </span>
                    </td>
                  )
                })}
                <td colSpan={2} className="px-3 py-2 text-right">
                  <span className="inline-flex items-center gap-1 rounded border border-rose-500/45 bg-rose-500/15 px-2 py-0.5 font-bold text-rose-200">
                    <ShieldAlert className="h-3 w-3" />
                    MOST LETHAL: {columnAverages.mostLethal.labelEn.toUpperCase()} (
                    {columnAverages.mostLethal.avg.toFixed(1)}%)
                  </span>
                </td>
              </tr>
            </tfoot>
          )}
        </table>
      </div>
    </section>
  )
}
