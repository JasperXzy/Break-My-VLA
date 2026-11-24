import type { Precision } from './protocol'

export interface RadarScores {
  /** Level 0 nominal / clean success rate (0..100) */
  clean: number
  /** Physical chaos resilience: object_nudge, arm_kick, friction_glitch (0..100) */
  physical: number
  /** Visual chaos resilience: lighting_shift, camera_shake, occlusion_patch (0..100) */
  visual: number
  /** Systems chaos resilience: latency_inject, wrist_blackout (0..100) */
  systems: number
  /** Brain downgrade resilience: quantization & low denoise steps (0..100) */
  quantization: number
}

export const RADAR_AXES: Array<{
  key: keyof RadarScores
  labelZh: string
  labelEn: string
  short: string
}> = [
  { key: 'clean', labelZh: '基准任务', labelEn: 'Clean', short: 'CLEAN' },
  { key: 'physical', labelZh: '物理抗扰', labelEn: 'Physical', short: 'PHYS' },
  { key: 'visual', labelZh: '视觉抗扰', labelEn: 'Visual', short: 'VIS' },
  { key: 'systems', labelZh: '系统延迟', labelEn: 'Systems', short: 'SYS' },
  { key: 'quantization', labelZh: '量化降噪', labelEn: 'Quant/Denoise', short: 'QUANT' },
]

export type InferenceBackend = 'pytorch' | 'tensorrt'

export interface AlignmentMetrics {
  /** Mean squared error of quantized/TRT action chunks vs BF16 PyTorch reference */
  action_mse_vs_bf16?: number
  /** Mean absolute (L1) error of action chunks vs BF16 PyTorch reference */
  action_l1_vs_bf16?: number
  /** Cosine similarity of action chunks vs BF16 PyTorch reference (-1..1) */
  action_cos_vs_bf16?: number
}

export interface LatencyMetrics {
  mean_ms: number
  p50_ms: number
  p95_ms: number
  p99_ms?: number
  fps: number
}

export interface BenchmarkEntry {
  id: string
  model: string
  family: string
  backend?: InferenceBackend
  precision: Precision
  denoise_steps: number
  params_b: number
  vram_gb: number
  overall_score: number
  latency: LatencyMetrics
  radar_scores: RadarScores
  alignment?: AlignmentMetrics
}

export function formatBackendLabel(backend?: InferenceBackend): string {
  if (backend === 'tensorrt') return 'TensorRT'
  if (backend === 'pytorch') return 'PyTorch'
  return 'PyTorch'
}

export interface ReplayComparison {
  id: string
  title: string
  chaos_tag: string
  prompt: string
  baseline_label: string
  baseline_precision: Precision
  baseline_latency_ms: number
  baseline_gif: string
  baseline_outcome: 'pass' | 'fail'
  degraded_label: string
  degraded_precision: Precision
  degraded_latency_ms: number
  degraded_gif: string
  degraded_outcome: 'pass' | 'fail'
  failure_note: string
}

export type DatasetVerification = 'real_gpu' | 'showcase' | 'cpu_mock'

export interface BenchmarkDataset {
  fileName: string
  sourcePath: string
  schema_version: string
  suite_name: string
  generated_at: string
  hardware: string
  notes: string
  verification: DatasetVerification
  entries: BenchmarkEntry[]
  replays: ReplayComparison[]
}

const VERIFICATION_PRIORITY: Record<DatasetVerification, number> = {
  real_gpu: 0,
  showcase: 1,
  cpu_mock: 2,
}

export function classifyBenchmarkDataset(
  meta: {
    fileName: string
    suite_name: string
    hardware: string
    notes: string
    verified?: unknown
    provenance?: unknown
  },
): DatasetVerification {
  const explicit = String(meta.provenance ?? '').toLowerCase()
  if (explicit === 'real_gpu' || explicit === 'real' || meta.verified === true) {
    return 'real_gpu'
  }
  if (explicit === 'cpu_mock' || explicit === 'mock') {
    return 'cpu_mock'
  }
  if (explicit === 'showcase' || explicit === 'reference') {
    return 'showcase'
  }

  const hay = `${meta.fileName} ${meta.suite_name} ${meta.hardware} ${meta.notes}`.toLowerCase()
  const fileAndSuite = `${meta.fileName} ${meta.suite_name}`.toLowerCase()

  if (fileAndSuite.includes('mock') || meta.hardware.toLowerCase().includes('mock')) {
    return 'cpu_mock'
  }
  if (
    fileAndSuite.includes('showcase') ||
    meta.hardware.toLowerCase().includes('preview') ||
    meta.notes.toLowerCase().includes('static reference')
  ) {
    return 'showcase'
  }
  if (
    fileAndSuite.includes('real') ||
    hay.includes('measured closed-loop') ||
    hay.includes('real gpu') ||
    /\bnvidia\s+(l4|a100|h100|h200|rtx|t4|g4)\b/i.test(meta.hardware)
  ) {
    return 'real_gpu'
  }
  return 'showcase'
}

export function sortBenchmarkDatasets(datasets: BenchmarkDataset[]): BenchmarkDataset[] {
  return [...datasets].sort((a, b) => {
    const rankDiff = VERIFICATION_PRIORITY[a.verification] - VERIFICATION_PRIORITY[b.verification]
    if (rankDiff !== 0) return rankDiff
    return a.fileName.localeCompare(b.fileName)
  })
}

const VALID_PRECISIONS = new Set<Precision>(['bf16', 'fp8', 'int8', 'nvfp4'])

function toPrecision(raw: unknown): Precision {
  const s = String(raw ?? 'bf16').toLowerCase() as Precision
  return VALID_PRECISIONS.has(s) ? s : 'bf16'
}

/** Normalizes either 0..1 ratios or 0..100 percentages to 0..100. */
function toScore100(raw: unknown, fallback = 0): number {
  const n = Number(raw)
  if (!Number.isFinite(n)) return fallback
  const scaled = n > 0 && n <= 1 ? n * 100 : n
  return Number(Math.max(0, Math.min(100, scaled)).toFixed(1))
}

function toFinite(raw: unknown, fallback = 0): number {
  const n = Number(raw)
  return Number.isFinite(n) ? Number(n.toFixed(2)) : fallback
}

function toOptionalFinite(raw: unknown, digits = 6, min?: number, max?: number): number | undefined {
  if (raw === undefined || raw === null || raw === '') return undefined
  const n = Number(raw)
  if (!Number.isFinite(n)) return undefined
  let clamped = n
  if (min !== undefined) clamped = Math.max(min, clamped)
  if (max !== undefined) clamped = Math.min(max, clamped)
  return Number(clamped.toFixed(digits))
}

function toBackend(raw: unknown): InferenceBackend | undefined {
  if (typeof raw !== 'string') return undefined
  const norm = raw.trim().toLowerCase()
  if (norm === 'tensorrt' || norm === 'trt' || norm === 'tensorrt-llm' || norm === 'trt-llm') {
    return 'tensorrt'
  }
  if (norm === 'pytorch' || norm === 'torch' || norm === 'eager') {
    return 'pytorch'
  }
  return undefined
}

function toAlignment(raw: unknown): AlignmentMetrics | undefined {
  if (!raw || typeof raw !== 'object') return undefined
  const obj = raw as Record<string, unknown>
  const mse = toOptionalFinite(obj.action_mse_vs_bf16, 6, 0)
  const l1 = toOptionalFinite(obj.action_l1_vs_bf16, 6, 0)
  const cos = toOptionalFinite(obj.action_cos_vs_bf16, 4, -1, 1)
  if (mse === undefined && l1 === undefined && cos === undefined) return undefined
  const result: AlignmentMetrics = {}
  if (mse !== undefined) result.action_mse_vs_bf16 = mse
  if (l1 !== undefined) result.action_l1_vs_bf16 = l1
  if (cos !== undefined) result.action_cos_vs_bf16 = cos
  return result
}

export function normalizeBenchmarkDataset(
  raw: unknown,
  fileName: string,
  sourcePath: string,
): BenchmarkDataset {
  const obj = (raw && typeof raw === 'object' ? raw : {}) as Record<string, unknown>
  const rawEntries = Array.isArray(obj.entries) ? obj.entries : []
  const rawReplays = Array.isArray(obj.replays) ? obj.replays : []

  const entries: BenchmarkEntry[] = rawEntries.map((item, idx) => {
    const e = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
    const rawRadar = (e.radar_scores && typeof e.radar_scores === 'object'
      ? e.radar_scores
      : {}) as Record<string, unknown>
    const rawLat = (e.latency && typeof e.latency === 'object'
      ? e.latency
      : {}) as Record<string, unknown>

    const radar_scores: RadarScores = {
      clean: toScore100(rawRadar.clean, 0),
      physical: toScore100(rawRadar.physical, 0),
      visual: toScore100(rawRadar.visual, 0),
      systems: toScore100(rawRadar.systems, 0),
      quantization: toScore100(rawRadar.quantization, 0),
    }

    const avgRadar =
      (radar_scores.clean +
        radar_scores.physical +
        radar_scores.visual +
        radar_scores.systems +
        radar_scores.quantization) /
      5

    const meanMs = toFinite(rawLat.mean_ms ?? e.infer_ms, 50)
    const p50Ms = toFinite(rawLat.p50_ms, meanMs * 0.96)
    const p95Ms = toFinite(rawLat.p95_ms, meanMs * 1.18)
    const p99Ms = toOptionalFinite(rawLat.p99_ms, 2, 0)
    const fps = toFinite(rawLat.fps, meanMs > 0 ? 1000 / meanMs : 0)

    const precision = toPrecision(e.precision)
    const model = String(e.model ?? `model-${idx + 1}`)
    const backend = toBackend(e.backend)
    const alignment = toAlignment(e.alignment)

    const latency: LatencyMetrics = {
      mean_ms: meanMs,
      p50_ms: p50Ms,
      p95_ms: p95Ms,
      fps,
    }
    if (p99Ms !== undefined) {
      latency.p99_ms = p99Ms
    }

    const entry: BenchmarkEntry = {
      id: String(e.id ?? `${model}-${precision}-${idx}`),
      model,
      family: String(e.family ?? 'VLA Policy'),
      precision,
      denoise_steps: Math.max(1, Math.min(10, Math.round(Number(e.denoise_steps ?? 8)))),
      params_b: toFinite(e.params_b, 0),
      vram_gb: toFinite(e.vram_gb, 0),
      overall_score:
        e.overall_score !== undefined ? toScore100(e.overall_score) : Number(avgRadar.toFixed(1)),
      latency,
      radar_scores,
    }
    if (backend !== undefined) {
      entry.backend = backend
    }
    if (alignment !== undefined) {
      entry.alignment = alignment
    }
    return entry
  })

  const replays: ReplayComparison[] = rawReplays.map((item, idx) => {
    const r = (item && typeof item === 'object' ? item : {}) as Record<string, unknown>
    return {
      id: String(r.id ?? `replay-${idx + 1}`),
      title: String(r.title ?? `Chaos Comparison #${idx + 1}`),
      chaos_tag: String(r.chaos_tag ?? 'object_nudge'),
      prompt: String(r.prompt ?? 'pick up the crimson block and place it in the cyan bowl'),
      baseline_label: String(r.baseline_label ?? 'Baseline BF16'),
      baseline_precision: toPrecision(r.baseline_precision ?? 'bf16'),
      baseline_latency_ms: toFinite(r.baseline_latency_ms, 50),
      baseline_gif: String(r.baseline_gif ?? './replays/bf16_clean_pass.gif'),
      baseline_outcome: r.baseline_outcome === 'fail' ? 'fail' : 'pass',
      degraded_label: String(r.degraded_label ?? 'Degraded NVFP4'),
      degraded_precision: toPrecision(r.degraded_precision ?? 'nvfp4'),
      degraded_latency_ms: toFinite(r.degraded_latency_ms, 18),
      degraded_gif: String(r.degraded_gif ?? './replays/nvfp4_nudge_fail.gif'),
      degraded_outcome: r.degraded_outcome === 'pass' ? 'pass' : 'fail',
      failure_note: String(r.failure_note ?? ''),
    }
  })

  const suite_name = String(obj.suite_name ?? fileName.replace(/\.json$/i, ''))
  const hardware = String(obj.hardware ?? 'Unspecified Hardware')
  const notes = String(obj.notes ?? '')
  const verification = classifyBenchmarkDataset({
    fileName,
    suite_name,
    hardware,
    notes,
    verified: obj.verified,
    provenance: obj.provenance,
  })

  return {
    fileName,
    sourcePath,
    schema_version: String(obj.schema_version ?? '1.0'),
    suite_name,
    generated_at: String(obj.generated_at ?? ''),
    hardware,
    notes,
    verification,
    entries,
    replays,
  }
}
