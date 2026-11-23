import { useEffect, useMemo, useState } from 'react'
import {
  ArrowLeftRight,
  Award,
  Check,
  Copy,
  Download,
  Gauge,
  HardDrive,
  Swords,
  Trophy,
} from 'lucide-react'
import type { BenchmarkDataset, BenchmarkEntry } from '../types/benchmark'
import type { Precision } from '../types/protocol'
import {
  type ComparisonSide,
  computeHeadToHeadMetrics,
  downloadMarkdownFile,
  formatEntryTag,
  formatHeadToHeadMarkdown,
  formatSignedDelta,
} from '../utils/arenaMarkdown'

interface HeadToHeadCardProps {
  dataset: BenchmarkDataset
  entries: BenchmarkEntry[]
  contenderAId: string
  contenderBId: string
  onChangeContenderA: (id: string) => void
  onChangeContenderB: (id: string) => void
  onSwapContenders: () => void
}

const PRECISION_PILL: Record<Precision, string> = {
  bf16: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-300',
  fp8: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-300',
  int8: 'border-amber-400/50 bg-amber-500/15 text-amber-300',
  nvfp4: 'border-fuchsia-400/50 bg-fuchsia-500/15 text-fuchsia-300',
}

function winnerPill(side: ComparisonSide): { text: string; cls: string } {
  if (side === 'a') {
    return {
      text: 'A WINS',
      cls: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200',
    }
  }
  if (side === 'b') {
    return {
      text: 'B WINS',
      cls: 'border-fuchsia-400/50 bg-fuchsia-500/15 text-fuchsia-200',
    }
  }
  return {
    text: 'TIE',
    cls: 'border-slate-600/70 bg-slate-800/80 text-slate-300',
  }
}

export function HeadToHeadCard({
  dataset,
  entries,
  contenderAId,
  contenderBId,
  onChangeContenderA,
  onChangeContenderB,
  onSwapContenders,
}: HeadToHeadCardProps) {
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    if (!copied) return
    const timer = window.setTimeout(() => setCopied(false), 2000)
    return () => window.clearTimeout(timer)
  }, [copied])

  const contenderA = useMemo(
    () => entries.find((e) => e.id === contenderAId) ?? entries[0] ?? null,
    [contenderAId, entries],
  )
  const contenderB = useMemo(
    () =>
      entries.find((e) => e.id === contenderBId) ??
      entries[Math.min(1, entries.length - 1)] ??
      entries[0] ??
      null,
    [contenderBId, entries],
  )

  const metrics = useMemo(() => {
    if (!contenderA || !contenderB) return null
    return computeHeadToHeadMetrics(contenderA, contenderB)
  }, [contenderA, contenderB])

  const markdownTable = useMemo(() => {
    if (!contenderA || !contenderB) return ''
    return formatHeadToHeadMarkdown(dataset, contenderA, contenderB)
  }, [contenderA, contenderB, dataset])

  if (!contenderA || !contenderB || !metrics) {
    return null
  }

  const handleCopyMarkdown = async () => {
    if (!markdownTable) return
    try {
      if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
        await navigator.clipboard.writeText(markdownTable)
      } else {
        const ta = document.createElement('textarea')
        ta.value = markdownTable
        ta.style.position = 'fixed'
        ta.style.opacity = '0'
        document.body.appendChild(ta)
        ta.select()
        document.execCommand('copy')
        document.body.removeChild(ta)
      }
      setCopied(true)
    } catch {
      setCopied(false)
    }
  }

  const handleExportMarkdown = () => {
    if (!markdownTable) return
    const slugA = contenderA.id.replace(/[^a-zA-Z0-9_-]/g, '_')
    const slugB = contenderB.id.replace(/[^a-zA-Z0-9_-]/g, '_')
    downloadMarkdownFile(markdownTable, `arena-h2h-${slugA}-vs-${slugB}.md`)
  }

  const overallBadge = winnerPill(metrics.overallWinner)
  const speedBadge = winnerPill(metrics.fasterSide)
  const vramBadge = winnerPill(metrics.vramWinner)

  return (
    <section
      aria-label="Head-to-Head 1v1 Comparison Card"
      className="arcade-panel rounded-xl p-3.5"
    >
      {/* Header + Copy/Export Markdown Controls */}
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2 border-b border-slate-800/90 pb-2.5">
        <div>
          <div className="flex items-center gap-2 font-mono text-xs font-bold tracking-wider text-fuchsia-300">
            <Swords className="h-4 w-4 text-cyan-300" />
            <span>HEAD-TO-HEAD 1V1 ARENA // 模型×精度 双雄对决卡</span>
          </div>
          <p className="font-mono text-[10px] text-slate-400">
            任选榜单中任意 2 个模型×精度组合，同屏对比五维抗压差值 (Δ)、推理加速比与显存节省
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-1.5">
          <button
            type="button"
            onClick={onSwapContenders}
            className="flex items-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-slate-200 transition hover:border-cyan-400/60 hover:text-white active:scale-95"
            title="Swap Contender A and Contender B"
          >
            <ArrowLeftRight className="h-3.5 w-3.5 text-cyan-300" />
            <span>SWAP A↔B</span>
          </button>

          <button
            type="button"
            onClick={() => void handleCopyMarkdown()}
            className={`flex items-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] font-semibold transition active:scale-95 ${
              copied
                ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                : 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200 hover:bg-cyan-500/25'
            }`}
          >
            {copied ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copied ? 'COPIED MD!' : 'COPY MARKDOWN'}</span>
          </button>

          <button
            type="button"
            onClick={handleExportMarkdown}
            className="flex items-center gap-1.5 rounded-lg border border-fuchsia-400/50 bg-fuchsia-500/15 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-fuchsia-200 transition hover:bg-fuchsia-500/25 active:scale-95"
          >
            <Download className="h-3.5 w-3.5" />
            <span>EXPORT .MD</span>
          </button>
        </div>
      </div>

      {/* Contender Pickers */}
      <div className="grid grid-cols-1 items-stretch gap-3 lg:grid-cols-11">
        {/* Contender A */}
        <div className="rounded-xl border border-cyan-500/40 bg-slate-950/85 p-3 lg:col-span-5">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-cyan-200">
              CONTENDER A · BASELINE
            </span>
            <span
              className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${PRECISION_PILL[contenderA.precision]}`}
            >
              {contenderA.precision} · {contenderA.denoise_steps}st
            </span>
          </div>
          <select
            aria-label="Select Contender A"
            value={contenderA.id}
            onChange={(e) => onChangeContenderA(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-900/95 px-2.5 py-1.5 font-mono text-xs font-semibold text-cyan-100 outline-none focus:border-cyan-400"
          >
            {entries.map((entry) => (
              <option key={entry.id} value={entry.id} className="bg-slate-950 text-slate-100">
                {formatEntryTag(entry)} — {entry.overall_score.toFixed(1)} pts ·{' '}
                {entry.latency.mean_ms.toFixed(1)}ms
              </option>
            ))}
          </select>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-slate-300">
            <span>
              Overall:{' '}
              <strong className="text-cyan-300">{contenderA.overall_score.toFixed(1)}</strong>
            </span>
            <span>
              Mean: <strong className="text-white">{contenderA.latency.mean_ms.toFixed(1)} ms</strong>{' '}
              ({contenderA.latency.fps.toFixed(1)} Hz)
            </span>
            <span>
              VRAM: <strong className="text-slate-100">{contenderA.vram_gb.toFixed(2)} GB</strong>
            </span>
          </div>
        </div>

        {/* VS Divider */}
        <div className="flex items-center justify-center lg:col-span-1">
          <div className="flex h-10 w-10 items-center justify-center rounded-full border border-fuchsia-400/50 bg-gradient-to-br from-cyan-500/20 to-fuchsia-500/25 font-mono text-xs font-extrabold tracking-widest text-white shadow-[0_0_18px_rgba(232,121,249,0.28)]">
            VS
          </div>
        </div>

        {/* Contender B */}
        <div className="rounded-xl border border-fuchsia-500/40 bg-slate-950/85 p-3 lg:col-span-5">
          <div className="mb-1.5 flex items-center justify-between gap-2">
            <span className="rounded bg-fuchsia-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-fuchsia-200">
              CONTENDER B · CHALLENGER
            </span>
            <span
              className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${PRECISION_PILL[contenderB.precision]}`}
            >
              {contenderB.precision} · {contenderB.denoise_steps}st
            </span>
          </div>
          <select
            aria-label="Select Contender B"
            value={contenderB.id}
            onChange={(e) => onChangeContenderB(e.target.value)}
            className="w-full rounded-lg border border-slate-700 bg-slate-900/95 px-2.5 py-1.5 font-mono text-xs font-semibold text-fuchsia-100 outline-none focus:border-fuchsia-400"
          >
            {entries.map((entry) => (
              <option key={entry.id} value={entry.id} className="bg-slate-950 text-slate-100">
                {formatEntryTag(entry)} — {entry.overall_score.toFixed(1)} pts ·{' '}
                {entry.latency.mean_ms.toFixed(1)}ms
              </option>
            ))}
          </select>
          <div className="mt-2 flex flex-wrap items-center justify-between gap-2 font-mono text-[11px] text-slate-300">
            <span>
              Overall:{' '}
              <strong className="text-fuchsia-300">{contenderB.overall_score.toFixed(1)}</strong>
            </span>
            <span>
              Mean: <strong className="text-white">{contenderB.latency.mean_ms.toFixed(1)} ms</strong>{' '}
              ({contenderB.latency.fps.toFixed(1)} Hz)
            </span>
            <span>
              VRAM: <strong className="text-slate-100">{contenderB.vram_gb.toFixed(2)} GB</strong>
            </span>
          </div>
        </div>
      </div>

      {/* 3 Core Delta KPIs: Speedup, VRAM Savings, Overall Δ */}
      <div className="mt-3 grid grid-cols-1 gap-2.5 md:grid-cols-3">
        <div className="rounded-lg border border-slate-800 bg-slate-950/85 p-2.5 font-mono">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <Gauge className="h-3.5 w-3.5 text-amber-300" />
              LATENCY SPEEDUP
            </span>
            <span className={`rounded border px-1.5 py-0 text-[9px] font-bold ${speedBadge.cls}`}>
              {speedBadge.text}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span
              data-testid="h2h-speedup"
              className="text-lg font-extrabold text-amber-300"
            >
              {metrics.speedupRatio.toFixed(2)}×
            </span>
            <span className="text-[11px] text-slate-400">
              p50 {metrics.p50SpeedupRatio.toFixed(2)}×
            </span>
          </div>
          <div className="mt-0.5 text-[10px] text-slate-400">
            Δ Mean {formatSignedDelta(metrics.latencyDeltaMs, ' ms')} · FPS{' '}
            {formatSignedDelta(metrics.fpsDelta, ' Hz')}
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/85 p-2.5 font-mono">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <HardDrive className="h-3.5 w-3.5 text-emerald-300" />
              VRAM SAVINGS
            </span>
            <span className={`rounded border px-1.5 py-0 text-[9px] font-bold ${vramBadge.cls}`}>
              {vramBadge.text}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span
              data-testid="h2h-vram-savings"
              className="text-lg font-extrabold text-emerald-300"
            >
              {metrics.vramWinner === 'tie'
                ? '0.00 GB'
                : `${formatSignedDelta(metrics.vramDeltaGb, ' GB', 2)}`}
            </span>
            {metrics.vramSavedPct !== null && (
              <span className="text-[11px] text-emerald-200/80">
                ({metrics.vramSavedPct.toFixed(1)}% saved)
              </span>
            )}
          </div>
          <div className="mt-0.5 text-[10px] text-slate-400">
            A {contenderA.vram_gb.toFixed(2)} GB vs B {contenderB.vram_gb.toFixed(2)} GB
          </div>
        </div>

        <div className="rounded-lg border border-slate-800 bg-slate-950/85 p-2.5 font-mono">
          <div className="flex items-center justify-between text-[10px] text-slate-400">
            <span className="flex items-center gap-1">
              <Trophy className="h-3.5 w-3.5 text-cyan-300" />
              OVERALL ROBUSTNESS Δ
            </span>
            <span className={`rounded border px-1.5 py-0 text-[9px] font-bold ${overallBadge.cls}`}>
              {overallBadge.text}
            </span>
          </div>
          <div className="mt-1 flex items-baseline gap-2">
            <span
              data-testid="h2h-overall-delta"
              className={`text-lg font-extrabold ${
                metrics.overallDelta > 0
                  ? 'text-emerald-300'
                  : metrics.overallDelta < 0
                    ? 'text-rose-300'
                    : 'text-slate-200'
              }`}
            >
              {formatSignedDelta(metrics.overallDelta, ' pts')}
            </span>
            <span className="text-[11px] text-slate-400">
              5-Axis: A {metrics.axisWinsA} · B {metrics.axisWinsB} · Tie {metrics.axisTies}
            </span>
          </div>
          <div className="mt-0.5 text-[10px] text-slate-400">
            {contenderA.overall_score.toFixed(1)} → {contenderB.overall_score.toFixed(1)} / 100
          </div>
        </div>
      </div>

      {/* 5-Axis Robustness Delta Breakdown */}
      <div className="mt-3 grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-5">
        {metrics.axisDeltas.map((axis) => {
          const pill = winnerPill(axis.winner)
          const deltaColor =
            axis.delta > 0
              ? 'text-emerald-300'
              : axis.delta < 0
                ? 'text-rose-300'
                : 'text-slate-400'
          return (
            <div
              key={axis.key}
              className="rounded-lg border border-slate-800/90 bg-slate-950/75 p-2.5 font-mono"
            >
              <div className="flex items-center justify-between gap-1 text-[10px]">
                <span className="font-bold text-slate-200">
                  {axis.labelEn}{' '}
                  <span className="font-normal text-slate-500">· {axis.labelZh}</span>
                </span>
                <span className={`rounded border px-1 py-0 text-[9px] font-bold ${pill.cls}`}>
                  {axis.winner === 'tie' ? 'TIE' : axis.winner.toUpperCase()}
                </span>
              </div>

              <div className="mt-1.5 flex items-baseline justify-between">
                <span className="text-[11px] text-cyan-300">A {axis.scoreA.toFixed(0)}</span>
                <span className={`text-xs font-extrabold ${deltaColor}`}>
                  Δ {formatSignedDelta(axis.delta)}
                </span>
                <span className="text-[11px] text-fuchsia-300">B {axis.scoreB.toFixed(0)}</span>
              </div>

              {/* Dual comparison mini bars */}
              <div className="mt-1.5 space-y-1">
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900">
                  <div
                    className="h-full rounded-full bg-cyan-400"
                    style={{ width: `${Math.max(2, axis.scoreA)}%` }}
                  />
                </div>
                <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900">
                  <div
                    className="h-full rounded-full bg-fuchsia-400"
                    style={{ width: `${Math.max(2, axis.scoreB)}%` }}
                  />
                </div>
              </div>
            </div>
          )
        })}
      </div>

      {/* Winner Summary Banner */}
      <div className="mt-3 flex flex-wrap items-center justify-between gap-2 rounded-lg border border-cyan-500/30 bg-gradient-to-r from-cyan-950/40 via-slate-950/80 to-fuchsia-950/40 px-3 py-2 font-mono text-[11px]">
        <div className="flex items-center gap-2 text-slate-100">
          <Award className="h-4 w-4 shrink-0 text-amber-300" />
          <span className="font-bold text-amber-200">ARENA VERDICT:</span>
          <span data-testid="h2h-verdict">{metrics.summaryVerdict}</span>
        </div>
      </div>
    </section>
  )
}
