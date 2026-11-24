import {
  type BenchmarkDataset,
  type BenchmarkEntry,
  type DatasetVerification,
  formatBackendLabel,
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
  p99DeltaMs: number | null
  p99SpeedupRatio: number | null
  p99Winner: ComparisonSide
  cosDelta: number | null
  cosWinner: ComparisonSide
  mseDelta: number | null
  mseWinner: ComparisonSide
  hasM4Metrics: boolean
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
  const backendPart =
    entry.backend === 'tensorrt' ? ' · TRT' : entry.backend === 'pytorch' ? ' · PT' : ''
  return `${entry.model} (${entry.precision.toUpperCase()} · ${entry.denoise_steps}st${backendPart})`
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

export function formatAlignmentCosine(value?: number): string {
  if (value === undefined || !Number.isFinite(value)) return '—'
  return value.toFixed(4)
}

export function formatAlignmentError(value?: number): string {
  if (value === undefined || !Number.isFinite(value)) return '—'
  if (value === 0) return '0.0000'
  if (value < 0.0001) return value.toExponential(2)
  return value.toFixed(5)
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

  // Optional M4 P99 tail latency comparison
  const p99A = contenderA.latency.p99_ms
  const p99B = contenderB.latency.p99_ms
  let p99DeltaMs: number | null = null
  let p99SpeedupRatio: number | null = null
  let p99Winner: ComparisonSide = 'tie'
  if (p99A !== undefined && p99B !== undefined) {
    const safeA = Math.max(0.01, p99A)
    const safeB = Math.max(0.01, p99B)
    p99DeltaMs = Number((safeB - safeA).toFixed(1))
    p99Winner = Math.abs(p99DeltaMs) < 0.1 ? 'tie' : safeB < safeA ? 'b' : 'a'
    p99SpeedupRatio = Number(
      (p99Winner === 'b' ? safeA / safeB : p99Winner === 'a' ? safeB / safeA : 1).toFixed(2),
    )
  }

  // Optional M4 action-chunk alignment comparison (higher cosine is better; lower MSE is better)
  const cosA = contenderA.alignment?.action_cos_vs_bf16
  const cosB = contenderB.alignment?.action_cos_vs_bf16
  let cosDelta: number | null = null
  let cosWinner: ComparisonSide = 'tie'
  if (cosA !== undefined && cosB !== undefined) {
    cosDelta = Number((cosB - cosA).toFixed(4))
    cosWinner = Math.abs(cosDelta) < 0.0001 ? 'tie' : cosDelta > 0 ? 'b' : 'a'
  }

  const mseA = contenderA.alignment?.action_mse_vs_bf16
  const mseB = contenderB.alignment?.action_mse_vs_bf16
  let mseDelta: number | null = null
  let mseWinner: ComparisonSide = 'tie'
  if (mseA !== undefined && mseB !== undefined) {
    mseDelta = Number((mseB - mseA).toFixed(6))
    mseWinner = Math.abs(mseDelta) < 1e-6 ? 'tie' : mseB < mseA ? 'b' : 'a'
  }

  const hasM4Metrics = Boolean(
    contenderA.backend ||
      contenderB.backend ||
      p99A !== undefined ||
      p99B !== undefined ||
      contenderA.alignment ||
      contenderB.alignment,
  )

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
    p99DeltaMs,
    p99SpeedupRatio,
    p99Winner,
    cosDelta,
    cosWinner,
    mseDelta,
    mseWinner,
    hasM4Metrics,
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

  const backendA = contenderA.backend ? formatBackendLabel(contenderA.backend) : '—'
  const backendB = contenderB.backend ? formatBackendLabel(contenderB.backend) : '—'

  const summaryRow = (side: 'A' | 'B', entry: BenchmarkEntry): string => {
    const backendCell = entry.backend ? formatBackendLabel(entry.backend) : '—'
    const latStr =
      entry.latency.p99_ms !== undefined
        ? `${entry.latency.mean_ms.toFixed(1)} / ${entry.latency.p95_ms.toFixed(1)} / ${entry.latency.p99_ms.toFixed(1)} ms`
        : `${entry.latency.mean_ms.toFixed(1)} / ${entry.latency.p95_ms.toFixed(1)} ms`
    const cosCell = formatAlignmentCosine(entry.alignment?.action_cos_vs_bf16)
    const mseCell = formatAlignmentError(entry.alignment?.action_mse_vs_bf16)
    return `| **${side}** | ${entry.model} | ${backendCell} | ${entry.precision.toUpperCase()} (${entry.denoise_steps}st) | ${latStr} | ${cosCell} | ${mseCell} | **${entry.overall_score.toFixed(1)}** |`
  }

  const extraM4Rows: string[] = []
  if (contenderA.backend || contenderB.backend) {
    extraM4Rows.push(`| Inference Backend | ${backendA} | ${backendB} | — | — |`)
  }
  if (contenderA.latency.p99_ms !== undefined || contenderB.latency.p99_ms !== undefined) {
    const p99CellA =
      contenderA.latency.p99_ms !== undefined ? `${contenderA.latency.p99_ms.toFixed(1)} ms` : '—'
    const p99CellB =
      contenderB.latency.p99_ms !== undefined ? `${contenderB.latency.p99_ms.toFixed(1)} ms` : '—'
    const p99DeltaCell =
      m.p99DeltaMs !== null && m.p99SpeedupRatio !== null
        ? `${formatSignedDelta(m.p99DeltaMs, ' ms')} (${m.p99SpeedupRatio.toFixed(2)}×)`
        : '—'
    extraM4Rows.push(
      `| P99 Tail Latency | ${p99CellA} | ${p99CellB} | ${p99DeltaCell} | ${
        m.p99DeltaMs !== null ? sideName(m.p99Winner) : '—'
      } |`,
    )
  }
  if (
    contenderA.alignment?.action_cos_vs_bf16 !== undefined ||
    contenderB.alignment?.action_cos_vs_bf16 !== undefined
  ) {
    const cosDeltaCell =
      m.cosDelta !== null ? formatSignedDelta(m.cosDelta, '', 4) : '—'
    extraM4Rows.push(
      `| Action Cosine (vs BF16) | ${formatAlignmentCosine(contenderA.alignment?.action_cos_vs_bf16)} | ${formatAlignmentCosine(contenderB.alignment?.action_cos_vs_bf16)} | ${cosDeltaCell} | ${
        m.cosDelta !== null ? sideName(m.cosWinner) : '—'
      } |`,
    )
  }
  if (
    contenderA.alignment?.action_mse_vs_bf16 !== undefined ||
    contenderB.alignment?.action_mse_vs_bf16 !== undefined
  ) {
    const mseDeltaCell =
      m.mseDelta !== null ? formatSignedDelta(m.mseDelta, '', 5) : '—'
    extraM4Rows.push(
      `| Action MSE (vs BF16) | ${formatAlignmentError(contenderA.alignment?.action_mse_vs_bf16)} | ${formatAlignmentError(contenderB.alignment?.action_mse_vs_bf16)} | ${mseDeltaCell} | ${
        m.mseDelta !== null ? sideName(m.mseWinner) : '—'
      } |`,
    )
  }

  const rows: string[] = [
    `### Break-My-VLA Multi-VLA Arena: ${nameA} vs ${nameB}`,
    '',
    `- **Suite**: ${dataset.suite_name} (\`${verificationBadgeLabel(dataset.verification)}\`)`,
    `- **Hardware**: ${dataset.hardware}`,
    `- **Source**: \`${dataset.sourcePath}\``,
    '',
    `| Side | Model | Backend | Precision | Mean / P95 / P99 | Action Cosine (vs BF16) | Action MSE (vs BF16) | Overall |`,
    `| --- | --- | --- | --- | ---: | ---: | ---: | ---: |`,
    summaryRow('A', contenderA),
    summaryRow('B', contenderB),
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
    ...extraM4Rows,
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
