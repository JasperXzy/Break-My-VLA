import { useEffect, useMemo, useState } from 'react'
import {
  Activity,
  ArrowLeftRight,
  Award,
  Check,
  Copy,
  Download,
  Gauge,
  HardDrive,
  Link2,
  Swords,
  Timer,
  Trophy,
} from 'lucide-react'
import {
  type BenchmarkDataset,
  type BenchmarkEntry,
  formatBackendLabel,
} from '../types/benchmark'
import type { Precision } from '../types/protocol'
import { buildArenaShareHash, buildArenaShareUrl } from '../utils/arenaHash'
import {
  type ComparisonSide,
  computeHeadToHeadMetrics,
  downloadMarkdownFile,
  formatAlignmentCosine,
  formatAlignmentError,
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

function getDriftVisual(cos?: number, mse?: number): {
  driftPct: number
  barClass: string
  statusText: string
  statusClass: string
} {
  if (cos !== undefined) {
    // 1.000 => 0% drift, 0.90 => 100% full-scale visual drift bar
    const rawDriftPct = Math.max(0, (1 - cos) * 100)
    const barPct = Math.min(100, rawDriftPct * 10)
    if (rawDriftPct < 0.05) {
      return {
        driftPct: barPct,
        barClass: 'bg-emerald-400',
        statusText: 'BF16 REF',
        statusClass: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-200',
      }
    }
    if (cos >= 0.99) {
      return {
        driftPct: Math.max(6, barPct),
        barClass: 'bg-cyan-400',
        statusText: 'ALIGNED',
        statusClass: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200',
      }
    }
    if (cos >= 0.97) {
      return {
        driftPct: Math.max(15, barPct),
        barClass: 'bg-amber-400',
        statusText: 'MILD DRIFT',
        statusClass: 'border-amber-400/50 bg-amber-500/15 text-amber-200',
      }
    }
    return {
      driftPct: Math.max(30, barPct),
      barClass: 'bg-rose-400',
      statusText: 'HIGH DRIFT',
      statusClass: 'border-rose-400/60 bg-rose-500/20 text-rose-200',
    }
  }

  if (mse !== undefined) {
    const barPct = Math.min(100, (mse / 0.015) * 100)
    if (mse <= 0.0001) {
      return {
        driftPct: Math.max(2, barPct),
        barClass: 'bg-emerald-400',
        statusText: 'BF16 REF',
        statusClass: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-200',
      }
    }
    if (mse <= 0.002) {
      return {
        driftPct: Math.max(10, barPct),
        barClass: 'bg-cyan-400',
        statusText: 'ALIGNED',
        statusClass: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200',
      }
    }
    return {
      driftPct: Math.max(25, barPct),
      barClass: 'bg-rose-400',
      statusText: 'DRIFT',
      statusClass: 'border-rose-400/60 bg-rose-500/20 text-rose-200',
    }
  }

  return {
    driftPct: 0,
    barClass: 'bg-slate-600',
    statusText: 'N/A',
    statusClass: 'border-slate-700 bg-slate-900 text-slate-400',
  }
}

function rtSlaBadge(p99Ms: number): { label: string; cls: string } {
  if (p99Ms <= 33) {
    return {
      label: '≤33ms · 30Hz RT',
      cls: 'border-emerald-400/50 bg-emerald-500/15 text-emerald-200',
    }
  }
  if (p99Ms <= 50) {
    return {
      label: '≤50ms · 20Hz RT',
      cls: 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200',
    }
  }
  return {
    label: '>50ms TAIL',
    cls: 'border-rose-400/50 bg-rose-500/15 text-rose-200',
  }
}

async function copyTextToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      await navigator.clipboard.writeText(text)
    } else {
      const ta = document.createElement('textarea')
      ta.value = text
      ta.style.position = 'fixed'
      ta.style.opacity = '0'
      document.body.appendChild(ta)
      ta.select()
      document.execCommand('copy')
      document.body.removeChild(ta)
    }
    return true
  } catch {
    return false
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
  const [copiedMd, setCopiedMd] = useState(false)
  const [copiedLink, setCopiedLink] = useState(false)

  useEffect(() => {
    if (!copiedMd) return
    const timer = window.setTimeout(() => setCopiedMd(false), 2000)
    return () => window.clearTimeout(timer)
  }, [copiedMd])

  useEffect(() => {
    if (!copiedLink) return
    const timer = window.setTimeout(() => setCopiedLink(false), 2000)
    return () => window.clearTimeout(timer)
  }, [copiedLink])

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
    const ok = await copyTextToClipboard(markdownTable)
    setCopiedMd(ok)
  }

  const handleCopyShareLink = async () => {
    const shareParams = {
      dataset: dataset.fileName,
      a: contenderA.id,
      b: contenderB.id,
    }
    const shareHash = buildArenaShareHash(shareParams)
    if (typeof window !== 'undefined') {
      window.history.replaceState(null, '', shareHash)
    }
    const shareUrl = buildArenaShareUrl(shareParams)
    const ok = await copyTextToClipboard(shareUrl)
    setCopiedLink(ok)
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

  const hasAlignment = Boolean(
    contenderA.alignment?.action_cos_vs_bf16 !== undefined ||
      contenderB.alignment?.action_cos_vs_bf16 !== undefined ||
      contenderA.alignment?.action_mse_vs_bf16 !== undefined ||
      contenderB.alignment?.action_mse_vs_bf16 !== undefined,
  )
  const hasP99 =
    contenderA.latency.p99_ms !== undefined || contenderB.latency.p99_ms !== undefined

  const maxP99 = Math.max(
    60,
    contenderA.latency.p99_ms ?? 0,
    contenderB.latency.p99_ms ?? 0,
  )

  return (
    <section
      aria-label="Head-to-Head 1v1 Comparison Card"
      className="arcade-panel rounded-xl p-3 sm:p-3.5"
    >
      {/* Header + Action Controls */}
      <div className="mb-3 flex flex-col gap-2.5 border-b border-slate-800/90 pb-2.5 sm:flex-row sm:flex-wrap sm:items-center sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2 font-mono text-xs font-bold tracking-wider text-fuchsia-300">
            <Swords className="h-4 w-4 shrink-0 text-cyan-300" />
            <span className="truncate">HEAD-TO-HEAD 1V1 ARENA // 模型×精度 双雄对决卡</span>
          </div>
          <p className="mt-0.5 font-mono text-[10px] text-slate-400">
            任选任意 2 个模型×精度×后端组合，对比五维抗压 Δ、P99 尾延迟、动作精度漂移并一键深链分享
          </p>
        </div>

        <div className="grid grid-cols-2 gap-1.5 sm:flex sm:flex-wrap sm:items-center">
          <button
            type="button"
            onClick={onSwapContenders}
            className="flex items-center justify-center gap-1 rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-slate-200 transition hover:border-cyan-400/60 hover:text-white active:scale-95"
            title="Swap Contender A and Contender B"
          >
            <ArrowLeftRight className="h-3.5 w-3.5 text-cyan-300" />
            <span>SWAP A↔B</span>
          </button>

          <button
            type="button"
            onClick={() => void handleCopyShareLink()}
            className={`flex items-center justify-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] font-semibold transition active:scale-95 ${
              copiedLink
                ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                : 'border-amber-400/50 bg-amber-500/15 text-amber-200 hover:bg-amber-500/25'
            }`}
            title="Copy deep-link URL to this exact 1v1 matchup"
          >
            {copiedLink ? <Check className="h-3.5 w-3.5" /> : <Link2 className="h-3.5 w-3.5" />}
            <span>{copiedLink ? 'COPIED LINK!' : 'COPY SHARE LINK'}</span>
          </button>

          <button
            type="button"
            onClick={() => void handleCopyMarkdown()}
            className={`flex items-center justify-center gap-1.5 rounded-lg border px-2.5 py-1.5 font-mono text-[11px] font-semibold transition active:scale-95 ${
              copiedMd
                ? 'border-emerald-400/60 bg-emerald-500/20 text-emerald-200'
                : 'border-cyan-400/50 bg-cyan-500/15 text-cyan-200 hover:bg-cyan-500/25'
            }`}
          >
            {copiedMd ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
            <span>{copiedMd ? 'COPIED MD!' : 'COPY MARKDOWN'}</span>
          </button>

          <button
            type="button"
            onClick={handleExportMarkdown}
            className="flex items-center justify-center gap-1.5 rounded-lg border border-fuchsia-400/50 bg-fuchsia-500/15 px-2.5 py-1.5 font-mono text-[11px] font-semibold text-fuchsia-200 transition hover:bg-fuchsia-500/25 active:scale-95"
          >
            <Download className="h-3.5 w-3.5" />
            <span>EXPORT .MD</span>
          </button>
        </div>
      </div>

      {/* Contender Pickers */}
      <div className="grid grid-cols-1 items-stretch gap-3 lg:grid-cols-11">
        {/* Contender A */}
        <div className="min-w-0 rounded-xl border border-cyan-500/40 bg-slate-950/85 p-3 lg:col-span-5">
          <div className="mb-1.5 flex flex-wrap items-center justify-between gap-1.5">
            <span className="rounded bg-cyan-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-cyan-200">
              CONTENDER A · BASELINE
            </span>
            <div className="flex flex-wrap items-center gap-1">
              {contenderA.backend && (
                <span
                  data-testid="h2h-backend-a"
                  className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                    contenderA.backend === 'tensorrt'
                      ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-200'
                      : 'border-slate-600 bg-slate-800/80 text-slate-300'
                  }`}
                >
                  {contenderA.backend === 'tensorrt' ? '◆ TensorRT' : '● PyTorch'}
                </span>
              )}
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${PRECISION_PILL[contenderA.precision]}`}
              >
                {contenderA.precision} · {contenderA.denoise_steps}st
              </span>
            </div>
          </div>
          <select
            aria-label="Select Contender A"
            value={contenderA.id}
            onChange={(e) => onChangeContenderA(e.target.value)}
            className="w-full min-w-0 truncate rounded-lg border border-slate-700 bg-slate-900/95 px-2.5 py-1.5 font-mono text-xs font-semibold text-cyan-100 outline-none focus:border-cyan-400"
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
          <div className="flex h-9 w-9 items-center justify-center rounded-full border border-fuchsia-400/50 bg-gradient-to-br from-cyan-500/20 to-fuchsia-500/25 font-mono text-xs font-extrabold tracking-widest text-white shadow-[0_0_18px_rgba(232,121,249,0.28)] sm:h-10 sm:w-10">
            VS
          </div>
        </div>

        {/* Contender B */}
        <div className="min-w-0 rounded-xl border border-fuchsia-500/40 bg-slate-950/85 p-3 lg:col-span-5">
          <div className="mb-1.5 flex flex-wrap items-center justify-between gap-1.5">
            <span className="rounded bg-fuchsia-500/20 px-1.5 py-0.5 font-mono text-[10px] font-bold text-fuchsia-200">
              CONTENDER B · CHALLENGER
            </span>
            <div className="flex flex-wrap items-center gap-1">
              {contenderB.backend && (
                <span
                  data-testid="h2h-backend-b"
                  className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold ${
                    contenderB.backend === 'tensorrt'
                      ? 'border-cyan-400/60 bg-cyan-500/15 text-cyan-200'
                      : 'border-slate-600 bg-slate-800/80 text-slate-300'
                  }`}
                >
                  {contenderB.backend === 'tensorrt' ? '◆ TensorRT' : '● PyTorch'}
                </span>
              )}
              <span
                className={`rounded border px-1.5 py-0.5 font-mono text-[10px] font-bold uppercase ${PRECISION_PILL[contenderB.precision]}`}
              >
                {contenderB.precision} · {contenderB.denoise_steps}st
              </span>
            </div>
          </div>
          <select
            aria-label="Select Contender B"
            value={contenderB.id}
            onChange={(e) => onChangeContenderB(e.target.value)}
            className="w-full min-w-0 truncate rounded-lg border border-slate-700 bg-slate-900/95 px-2.5 py-1.5 font-mono text-xs font-semibold text-fuchsia-100 outline-none focus:border-fuchsia-400"
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

      {/* M4 Action-Chunk Precision Drift & P99 Tail Latency Strip */}
      {(hasAlignment || hasP99) && (
        <div
          data-testid="h2h-m4-drift-panel"
          className="mt-3 grid grid-cols-1 gap-2.5 lg:grid-cols-2"
        >
          {/* Action-Chunk Precision Drift */}
          {hasAlignment && (
            <div className="rounded-xl border border-cyan-500/30 bg-slate-950/85 p-3 font-mono">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-1.5">
                <span className="flex items-center gap-1.5 text-[11px] font-bold text-cyan-300">
                  <Activity className="h-3.5 w-3.5" />
                  ACTION-CHUNK PRECISION DRIFT (vs BF16)
                </span>
                {metrics.cosDelta !== null && (
                  <span
                    className={`rounded border px-1.5 py-0 text-[9px] font-bold ${winnerPill(metrics.cosWinner).cls}`}
                  >
                    Δ COS {formatSignedDelta(metrics.cosDelta, '', 4)}
                  </span>
                )}
              </div>

              <div className="space-y-2.5">
                {(
                  [
                    { side: 'A', entry: contenderA, accent: 'text-cyan-300' },
                    { side: 'B', entry: contenderB, accent: 'text-fuchsia-300' },
                  ] as const
                ).map(({ side, entry, accent }) => {
                  const cos = entry.alignment?.action_cos_vs_bf16
                  const mse = entry.alignment?.action_mse_vs_bf16
                  const l1 = entry.alignment?.action_l1_vs_bf16
                  const visual = getDriftVisual(cos, mse)
                  return (
                    <div key={side} className="space-y-1">
                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10px]">
                        <span className="flex items-center gap-1.5">
                          <strong className={accent}>{side}</strong>
                          <span className="text-slate-300">
                            {entry.model} ({entry.precision.toUpperCase()}
                            {entry.backend ? ` · ${formatBackendLabel(entry.backend)}` : ''})
                          </span>
                        </span>
                        <span
                          className={`rounded border px-1.5 py-0 text-[9px] font-bold ${visual.statusClass}`}
                        >
                          {visual.statusText}
                        </span>
                      </div>
                      <div className="flex flex-wrap items-center justify-between gap-2 text-[10px] text-slate-400">
                        <span>
                          Cosine:{' '}
                          <strong className="text-slate-100">{formatAlignmentCosine(cos)}</strong>
                        </span>
                        <span>
                          MSE:{' '}
                          <strong className="text-slate-100">{formatAlignmentError(mse)}</strong>
                        </span>
                        {l1 !== undefined && (
                          <span>
                            L1:{' '}
                            <strong className="text-slate-200">{formatAlignmentError(l1)}</strong>
                          </span>
                        )}
                      </div>
                      {/* Drift magnitude bar (0 = exact BF16 match, wider = larger quantization drift) */}
                      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-900">
                        <div
                          className={`h-full rounded-full transition-all ${visual.barClass}`}
                          style={{ width: `${Math.max(2, visual.driftPct)}%` }}
                        />
                      </div>
                    </div>
                  )
                })}
              </div>
            </div>
          )}

          {/* P99 Tail Latency Comparison */}
          {hasP99 && (
            <div className="rounded-xl border border-amber-500/30 bg-slate-950/85 p-3 font-mono">
              <div className="mb-2 flex flex-wrap items-center justify-between gap-1.5">
                <span className="flex items-center gap-1.5 text-[11px] font-bold text-amber-300">
                  <Timer className="h-3.5 w-3.5" />
                  P99 TAIL LATENCY COMPARISON
                </span>
                {metrics.p99SpeedupRatio !== null && (
                  <span
                    data-testid="h2h-p99-delta"
                    className={`rounded border px-1.5 py-0 text-[9px] font-bold ${winnerPill(metrics.p99Winner).cls}`}
                  >
                    {metrics.p99SpeedupRatio.toFixed(2)}× · Δ{' '}
                    {formatSignedDelta(metrics.p99DeltaMs ?? 0, ' ms')}
                  </span>
                )}
              </div>

              <div className="space-y-2.5">
                {(
                  [
                    {
                      side: 'A',
                      entry: contenderA,
                      accent: 'text-cyan-300',
                      barCls: 'bg-cyan-400',
                    },
                    {
                      side: 'B',
                      entry: contenderB,
                      accent: 'text-fuchsia-300',
                      barCls: 'bg-fuchsia-400',
                    },
                  ] as const
                ).map(({ side, entry, accent, barCls }) => {
                  const p99 = entry.latency.p99_ms
                  const pct = p99 !== undefined ? Math.min(100, (p99 / maxP99) * 100) : 0
                  const sla = p99 !== undefined ? rtSlaBadge(p99) : null
                  return (
                    <div key={side} className="space-y-1">
                      <div className="flex flex-wrap items-center justify-between gap-1 text-[10px]">
                        <span className="flex items-center gap-1.5">
                          <strong className={accent}>{side}</strong>
                          <span className="text-slate-300">
                            {entry.model}
                            {entry.backend ? ` (${formatBackendLabel(entry.backend)})` : ''}
                          </span>
                        </span>
                        <div className="flex items-center gap-1.5">
                          <strong className="text-white">
                            {p99 !== undefined ? `${p99.toFixed(1)} ms` : '—'}
                          </strong>
                          {sla && (
                            <span
                              className={`rounded border px-1.5 py-0 text-[9px] font-bold ${sla.cls}`}
                            >
                              {sla.label}
                            </span>
                          )}
                        </div>
                      </div>
                      <div className="relative h-2 w-full overflow-hidden rounded-full bg-slate-900">
                        <div
                          className="pointer-events-none absolute inset-y-0 z-10 border-l border-dashed border-rose-400/75"
                          style={{ left: `${Math.min(100, (33 / maxP99) * 100)}%` }}
                          title="33ms (30Hz RT budget)"
                        />
                        <div
                          className="pointer-events-none absolute inset-y-0 z-10 border-l border-dashed border-amber-400/65"
                          style={{ left: `${Math.min(100, (50 / maxP99) * 100)}%` }}
                          title="50ms (20Hz RT budget)"
                        />
                        {p99 !== undefined && (
                          <div
                            className={`h-full rounded-full ${barCls}`}
                            style={{ width: `${Math.max(3, pct)}%` }}
                          />
                        )}
                      </div>
                    </div>
                  )
                })}
              </div>
              <div className="mt-2 flex flex-wrap items-center justify-between gap-1 text-[9px] text-slate-500">
                <span>Dashed markers: 33ms (30Hz) · 50ms (20Hz) real-time control ceilings</span>
              </div>
            </div>
          )}
        </div>
      )}

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
        <div className="flex items-start gap-2 text-slate-100 sm:items-center">
          <Award className="mt-0.5 h-4 w-4 shrink-0 text-amber-300 sm:mt-0" />
          <span className="shrink-0 font-bold text-amber-200">ARENA VERDICT:</span>
          <span data-testid="h2h-verdict" className="break-words">
            {metrics.summaryVerdict}
          </span>
        </div>
      </div>
    </section>
  )
}
