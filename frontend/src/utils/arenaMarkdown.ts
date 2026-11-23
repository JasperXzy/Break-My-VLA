import {
  type BenchmarkDataset,
  type BenchmarkEntry,
  type DatasetVerification,
  RADAR_AXES,
  type RadarScores,
} from '../types/benchmark'
import { triggerBlobDownload } from './gifEncoder'

export type ComparisonSide = 'a' | 'b' | 'tie'

export interface AxisDeltaMetric {
  key: keyof RadarScores
  labelZh: string
  labelEn: string
  short: string
  scoreA: number
  scoreB: number
  delta: number
  winner: ComparisonSide
}

export interface HeadToHeadMetrics {
  axisDeltas: AxisDeltaMetric[]
  overallDelta: number
  overallWinner: ComparisonSide
  latencyDeltaMs: number
  speedupRatio: number
  p50SpeedupRatio: number
  fasterSide: ComparisonSide
  fpsDelta: number
  vramDeltaGb: number
  vramSavedGb: number
  vramSavedPct: number | null
  vramWinner: ComparisonSide
  axisWinsA: number
  axisWinsB: number
  axisTies: number
  summaryVerdict: string
}

export function formatEntryTag(entry: BenchmarkEntry): string {
  return `${entry.model} (${entry.precision.toUpperCase()} · ${entry.denoise_steps}st)`
}

export function verificationBadgeLabel(v: DatasetVerification): string {
  if (v === 'real_gpu') return 'REAL GPU VERIFIED'
  if (v === 'cpu_mock') return 'CPU MOCK BASELINE'
  return 'SHOWCASE / REFERENCE'
}

export function formatSignedDelta(value: number, suffix = '', digits = 1): string {
  const rounded = Number(value.toFixed(digits))
  if (Math.abs(rounded) < 1e-6) return `0${digits > 0 ? `.${'0'.repeat(digits)}` : ''}${suffix}`
  const sign = rounded > 0 ? '+' : ''
  return `${sign}${rounded.toFixed(digits)}${suffix}`
}

export function computeHeadToHeadMetrics(
  contenderA: BenchmarkEntry,
  contenderB: BenchmarkEntry,
): HeadToHeadMetrics {
  const axisDeltas: AxisDeltaMetric[] = RADAR_AXES.map((axis) => {
    const scoreA = contenderA.radar_scores[axis.key]
    const scoreB = contenderB.radar_scores[axis.key]
    const delta = Number((scoreB - scoreA).toFixed(1))
    const winner: ComparisonSide =
      Math.abs(delta) < 0.05 ? 'tie' : delta > 0 ? 'b' : 'a'
    return {
      key: axis.key,
      labelZh: axis.labelZh,
      labelEn: axis.labelEn,
      short: axis.short,
      scoreA,
      scoreB,
      delta,
      winner,
    }
  })

  const axisWinsA = axisDeltas.filter((d) => d.winner === 'a').length
  const axisWinsB = axisDeltas.filter((d) => d.winner === 'b').length
  const axisTies = axisDeltas.filter((d) => d.winner === 'tie').length

  const overallDelta = Number((contenderB.overall_score - contenderA.overall_score).toFixed(1))
  const overallWinner: ComparisonSide =
    Math.abs(overallDelta) < 0.05 ? 'tie' : overallDelta > 0 ? 'b' : 'a'

  const meanA = Math.max(0.01, contenderA.latency.mean_ms)
  const meanB = Math.max(0.01, contenderB.latency.mean_ms)
  const latencyDeltaMs = Number((meanB - meanA).toFixed(1))
  const fasterSide: ComparisonSide =
    Math.abs(latencyDeltaMs) < 0.1 ? 'tie' : meanB < meanA ? 'b' : 'a'
  const speedupRatio = Number(
    (fasterSide === 'b' ? meanA / meanB : fasterSide === 'a' ? meanB / meanA : 1).toFixed(2),
  )

  const p50A = Math.max(0.01, contenderA.latency.p50_ms)
  const p50B = Math.max(0.01, contenderB.latency.p50_ms)
  const p50SpeedupRatio = Number(
    (p50B <= p50A ? p50A / p50B : p50B / p50A).toFixed(2),
  )

  const fpsDelta = Number((contenderB.latency.fps - contenderA.latency.fps).toFixed(1))

  const vramA = contenderA.vram_gb
  const vramB = contenderB.vram_gb
  const vramDeltaGb = Number((vramB - vramA).toFixed(2))
  const vramSavedGb = Number(Math.abs(vramDeltaGb).toFixed(2))
  const vramWinner: ComparisonSide =
    Math.abs(vramDeltaGb) < 0.01 ? 'tie' : vramB < vramA ? 'b' : 'a'
  const higherVram = Math.max(vramA, vramB)
  const vramSavedPct =
    higherVram > 0 && vramSavedGb > 0
      ? Number(((vramSavedGb / higherVram) * 100).toFixed(1))
      : null

  const labelA = formatEntryTag(contenderA)
  const labelB = formatEntryTag(contenderB)
  const parts: string[] = []

  if (overallWinner === 'b') {
    parts.push(`${labelB} leads Overall (${formatSignedDelta(overallDelta)} pts)`)
  } else if (overallWinner === 'a') {
    parts.push(`${labelA} leads Overall (+${Math.abs(overallDelta).toFixed(1)} pts)`)
  } else {
    parts.push(`Overall score tied (${contenderA.overall_score.toFixed(1)})`)
  }

  if (fasterSide === 'b') {
    parts.push(`${labelB} is ${speedupRatio.toFixed(2)}× faster`)
  } else if (fasterSide === 'a') {
    parts.push(`${labelA} is ${speedupRatio.toFixed(2)}× faster`)
  }

  if (vramWinner !== 'tie' && vramSavedPct !== null) {
    const winnerLabel = vramWinner === 'b' ? labelB : labelA
    parts.push(`${winnerLabel} saves ${vramSavedGb.toFixed(2)} GB VRAM (-${vramSavedPct.toFixed(1)}%)`)
  }

  return {
    axisDeltas,
    overallDelta,
    overallWinner,
    latencyDeltaMs,
    speedupRatio,
    p50SpeedupRatio,
    fasterSide,
    fpsDelta,
    vramDeltaGb,
    vramSavedGb,
    vramSavedPct,
    vramWinner,
    axisWinsA,
    axisWinsB,
    axisTies,
    summaryVerdict: parts.join(' · '),
  }
}

export function formatHeadToHeadMarkdown(
  dataset: BenchmarkDataset,
  contenderA: BenchmarkEntry,
  contenderB: BenchmarkEntry,
): string {
  const m = computeHeadToHeadMetrics(contenderA, contenderB)
  const nameA = formatEntryTag(contenderA)
  const nameB = formatEntryTag(contenderB)

  const sideName = (side: ComparisonSide): string => {
    if (side === 'a') return `**A** (${contenderA.precision.toUpperCase()})`
    if (side === 'b') return `**B** (${contenderB.precision.toUpperCase()})`
    return 'Tie'
  }

  const speedupNote =
    m.fasterSide === 'tie'
      ? '1.00×'
      : `${m.speedupRatio.toFixed(2)}× faster (${m.fasterSide.toUpperCase()})`

  const vramDeltaCell =
    m.vramWinner === 'tie'
      ? '0.00 GB'
      : `${formatSignedDelta(m.vramDeltaGb, ' GB', 2)}${
          m.vramSavedPct !== null ? ` (${m.vramSavedPct.toFixed(1)}% saved)` : ''
        }`

  const rows: string[] = [
    `### Break-My-VLA Multi-VLA Arena: ${nameA} vs ${nameB}`,
    '',
    `- **Suite**: ${dataset.suite_name} (\`${verificationBadgeLabel(dataset.verification)}\`)`,
    `- **Hardware**: ${dataset.hardware}`,
    `- **Source**: \`${dataset.sourcePath}\``,
    '',
    `| Metric | A: ${nameA} | B: ${nameB} | Delta (B − A) | Winner |`,
    `| --- | ---: | ---: | ---: | --- |`,
    `| **Overall Resilience** | ${contenderA.overall_score.toFixed(1)} | ${contenderB.overall_score.toFixed(1)} | ${formatSignedDelta(m.overallDelta)} | ${sideName(m.overallWinner)} |`,
    ...m.axisDeltas.map(
      (ax) =>
        `| ${ax.labelEn} (${ax.labelZh}) | ${ax.scoreA.toFixed(1)} | ${ax.scoreB.toFixed(1)} | ${formatSignedDelta(ax.delta)} | ${sideName(ax.winner)} |`,
    ),
    `| Mean Latency | ${contenderA.latency.mean_ms.toFixed(1)} ms | ${contenderB.latency.mean_ms.toFixed(1)} ms | ${formatSignedDelta(m.latencyDeltaMs, ' ms')} (${speedupNote}) | ${sideName(m.fasterSide)} |`,
    `| P50 / P95 Latency | ${contenderA.latency.p50_ms.toFixed(1)} / ${contenderA.latency.p95_ms.toFixed(1)} ms | ${contenderB.latency.p50_ms.toFixed(1)} / ${contenderB.latency.p95_ms.toFixed(1)} ms | p50 ${m.p50SpeedupRatio.toFixed(2)}× | ${sideName(m.fasterSide)} |`,
    `| Control Rate (FPS) | ${contenderA.latency.fps.toFixed(1)} Hz | ${contenderB.latency.fps.toFixed(1)} Hz | ${formatSignedDelta(m.fpsDelta, ' Hz')} | ${sideName(m.fasterSide)} |`,
    `| Peak VRAM | ${contenderA.vram_gb.toFixed(2)} GB | ${contenderB.vram_gb.toFixed(2)} GB | ${vramDeltaCell} | ${sideName(m.vramWinner)} |`,
    '',
    `> **Verdict**: ${m.summaryVerdict}`,
    '',
  ]

  return rows.join('\n')
}

export function downloadMarkdownFile(markdown: string, fileName: string): void {
  const blob = new Blob([markdown], { type: 'text/markdown;charset=utf-8' })
  triggerBlobDownload(blob, fileName)
}
