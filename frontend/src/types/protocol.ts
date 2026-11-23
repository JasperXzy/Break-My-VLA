/**
 * Strict wire protocol types aligned with backend/shared/schemas.py and SPEC.md.
 */

export type Precision = 'bf16' | 'fp8' | 'int8' | 'nvfp4'

export type ChaosSkill =
  | 'object_nudge'
  | 'arm_kick'
  | 'friction_glitch'
  | 'lighting_shift'
  | 'camera_shake'
  | 'occlusion_patch'
  | 'latency_inject'
  | 'wrist_blackout'

export type FrameStatus = 'running' | 'paused' | 'success' | 'policy_error'

/** 7-DoF LIBERO action: [dx, dy, dz, rx, ry, rz, gripper] */
export type Action7D = [number, number, number, number, number, number, number]

export type ActionChunk = Action7D[]

export interface FrameMessage {
  type: 'frame'
  step: number
  fps: number
  infer_ms: number
  precision: Precision
  denoise_steps: number
  rgb_main_b64: string
  rgb_wrist_b64: string
  action_chunk: ActionChunk
  active_chaos: ChaosSkill[]
  task_progress: number
  status: FrameStatus
  model?: string
}

export interface GifMessage {
  type: 'gif'
  gif_b64: string
}

export interface ErrorMessage {
  type: 'error'
  code: 'invalid_message' | 'gif_unavailable'
  message: string
}

export type ServerMessage = FrameMessage | GifMessage | ErrorMessage

export interface ObjectNudgeParams {
  dx: number
  dy: number
  dz: number
}

export interface ArmKickParams {
  dx: number
  dy: number
  dz: number
}

export interface FrictionGlitchParams {
  duration_steps: number
  multiplier: number
}

export interface LightingShiftParams {
  duration_steps: number
  brightness: number
  saturation: number
}

export interface CameraShakeParams {
  duration_steps: number
  amplitude_px: number
}

export interface OcclusionPatchParams {
  duration_steps: number
  x: number
  y: number
  width: number
  height: number
}

export interface LatencyInjectParams {
  duration_steps: number
  latency_ms: number
}

export interface WristBlackoutParams {
  duration_steps: number
}

export interface ChaosParamsMap {
  object_nudge: ObjectNudgeParams
  arm_kick: ArmKickParams
  friction_glitch: FrictionGlitchParams
  lighting_shift: LightingShiftParams
  camera_shake: CameraShakeParams
  occlusion_patch: OcclusionPatchParams
  latency_inject: LatencyInjectParams
  wrist_blackout: WristBlackoutParams
}

export type ChaosMessage<K extends ChaosSkill = ChaosSkill> = {
  type: 'chaos'
  skill: K
  params: ChaosParamsMap[K]
}

export interface ConfigMessage {
  type: 'config'
  model?: string
  precision?: Precision
  denoise_steps?: number
  latency_ms?: number
  prompt?: string
}

export interface VlaModelSpec {
  id: string
  label: string
  family: string
  paramsLabel: string
  paramsB?: number
  baselineLatencyMs: number
  precisions?: Precision[]
  description?: string
  badge?: string
}

export const DEMO_MODEL_PRESETS: VlaModelSpec[] = [
  {
    id: 'smolvla-450m',
    label: 'SmolVLA 450M',
    family: 'LeRobot Compact VLA',
    paramsLabel: '450M',
    paramsB: 0.45,
    baselineLatencyMs: 24,
    precisions: ['bf16', 'fp8', 'int8', 'nvfp4'],
    description: 'Compact 450M edge VLA: ultra-low latency, higher sensitivity to physical impulse & 4-bit noise',
    badge: 'COMPACT',
  },
  {
    id: 'pi0.5-libero',
    label: 'Pi0.5',
    family: 'OpenPI Flow-Matching',
    paramsLabel: '3.0B',
    paramsB: 3.0,
    baselineLatencyMs: 58,
    precisions: ['bf16', 'fp8', 'int8', 'nvfp4'],
    description: '3B flow-matching generalist with strong wrist-blackout recovery & smooth 16-step chunking',
    badge: 'BALANCED',
  },
  {
    id: 'isaac-gr00t-n1',
    label: 'Isaac GR00T',
    family: 'NVIDIA GR00T N1.5',
    paramsLabel: '2.2B',
    paramsB: 2.2,
    baselineLatencyMs: 42,
    precisions: ['bf16', 'fp8', 'int8', 'nvfp4'],
    description: '2.2B high-bandwidth visuomotor backbone with superior physical impulse damping',
    badge: 'CONTROL',
  },
  {
    id: 'openvla-oft-7b',
    label: 'OpenVLA-OFT',
    family: 'Parallel Action 7B',
    paramsLabel: '7.0B',
    paramsB: 7.0,
    baselineLatencyMs: 74,
    precisions: ['bf16', 'fp8', 'int8', 'nvfp4'],
    description: '7B semantic powerhouse with high visual chaos resilience and heavier compute baseline',
    badge: 'SEMANTIC',
  },
]

export interface ParsedHealthInfo {
  activeModel: string
  models: VlaModelSpec[]
  device?: string
  env?: string
}

function formatParamsLabel(paramsB?: number): string {
  if (typeof paramsB !== 'number' || !Number.isFinite(paramsB) || paramsB <= 0) {
    return 'N/A'
  }
  if (paramsB < 1) {
    return `${Math.round(paramsB * 1000)}M`
  }
  return `${paramsB.toFixed(1)}B`
}

function matchPresetByModelId(rawId: string): VlaModelSpec | undefined {
  const lower = rawId.toLowerCase()
  if (lower.includes('smolvla')) return DEMO_MODEL_PRESETS[0]
  if (lower.includes('pi0') || lower.includes('openpi')) return DEMO_MODEL_PRESETS[1]
  if (lower.includes('gr00t') || lower.includes('groot')) return DEMO_MODEL_PRESETS[2]
  if (lower.includes('openvla')) return DEMO_MODEL_PRESETS[3]
  return undefined
}

const VALID_PRECISIONS_SET = new Set<Precision>(['bf16', 'fp8', 'int8', 'nvfp4'])

function normalizePrecisions(raw: unknown): Precision[] | undefined {
  if (!Array.isArray(raw)) return undefined
  const out = raw
    .map((item) => String(item).toLowerCase() as Precision)
    .filter((item) => VALID_PRECISIONS_SET.has(item))
  return out.length > 0 ? out : undefined
}

export function buildModelSpecFromRaw(raw: unknown, fallbackPrecisions?: Precision[]): VlaModelSpec | null {
  if (typeof raw === 'string') {
    const id = raw.trim()
    if (!id) return null
    const preset = matchPresetByModelId(id)
    const isMock = id.toLowerCase().includes('mock')
    return {
      id,
      label: id,
      family: preset?.family ?? (isMock ? 'CPU Mock Policy' : 'Live VLA Policy'),
      paramsLabel: preset?.paramsLabel ?? (isMock ? '10M' : 'Custom'),
      paramsB: preset?.paramsB ?? (isMock ? 0.01 : undefined),
      baselineLatencyMs: preset?.baselineLatencyMs ?? (isMock ? 50 : 48),
      precisions: fallbackPrecisions ?? preset?.precisions,
      description: preset?.description,
      badge: preset?.badge ?? (isMock ? 'MOCK' : 'LIVE'),
    }
  }

  if (raw && typeof raw === 'object') {
    const obj = raw as Record<string, unknown>
    const id = String(obj.id ?? obj.model ?? obj.name ?? '').trim()
    if (!id) return null
    const preset = matchPresetByModelId(id)
    const isMock = id.toLowerCase().includes('mock')
    const paramsB =
      typeof obj.params_b === 'number' && Number.isFinite(obj.params_b)
        ? obj.params_b
        : preset?.paramsB
    const paramsLabel =
      typeof obj.params_label === 'string' && obj.params_label.trim()
        ? obj.params_label.trim()
        : typeof obj.params === 'string' && obj.params.trim()
          ? obj.params.trim()
          : paramsB !== undefined
            ? formatParamsLabel(paramsB)
            : preset?.paramsLabel ?? (isMock ? '10M' : 'Custom')
    const rawLatency = Number(obj.baseline_latency_ms ?? obj.latency_ms ?? obj.infer_ms)
    const baselineLatencyMs =
      Number.isFinite(rawLatency) && rawLatency > 0
        ? Math.round(rawLatency)
        : preset?.baselineLatencyMs ?? (isMock ? 50 : 48)

    return {
      id,
      label:
        typeof obj.label === 'string' && obj.label.trim()
          ? obj.label.trim()
          : typeof obj.display_name === 'string' && obj.display_name.trim()
            ? obj.display_name.trim()
            : id,
      family:
        typeof obj.family === 'string' && obj.family.trim()
          ? obj.family.trim()
          : preset?.family ?? (isMock ? 'CPU Mock Policy' : 'Live VLA Policy'),
      paramsLabel,
      paramsB,
      baselineLatencyMs,
      precisions: normalizePrecisions(obj.precisions) ?? fallbackPrecisions ?? preset?.precisions,
      description:
        typeof obj.description === 'string' ? obj.description : preset?.description,
      badge:
        typeof obj.badge === 'string' && obj.badge.trim()
          ? obj.badge.trim()
          : preset?.badge ?? (isMock ? 'MOCK' : 'LIVE'),
    }
  }

  return null
}

/**
 * Parses GET /health responses supporting both multi-model payloads
 * ({ active_model, models: [...] }) and legacy single-model payloads
 * ({ model, precisions, device, env }).
 */
export function parseHealthModels(raw: unknown): ParsedHealthInfo | null {
  if (!raw || typeof raw !== 'object') return null
  const obj = raw as Record<string, unknown>

  const topPrecisions = normalizePrecisions(obj.precisions)
  const rawActive = String(obj.active_model ?? obj.model ?? '').trim()
  const models: VlaModelSpec[] = []
  const seen = new Set<string>()

  const pushSpec = (spec: VlaModelSpec | null) => {
    if (!spec || seen.has(spec.id)) return
    seen.add(spec.id)
    models.push(spec)
  }

  if (Array.isArray(obj.models)) {
    for (const item of obj.models) {
      pushSpec(buildModelSpecFromRaw(item, topPrecisions))
    }
  } else if (obj.models && typeof obj.models === 'object') {
    for (const [key, value] of Object.entries(obj.models as Record<string, unknown>)) {
      if (value && typeof value === 'object') {
        pushSpec(buildModelSpecFromRaw({ id: key, ...(value as Record<string, unknown>) }, topPrecisions))
      } else {
        pushSpec(buildModelSpecFromRaw(key, topPrecisions))
      }
    }
  }

  if (rawActive && !seen.has(rawActive)) {
    const activeSpec = buildModelSpecFromRaw(
      {
        id: rawActive,
        params_b: obj.params_b,
        params: obj.params,
        baseline_latency_ms: obj.baseline_latency_ms,
        precisions: topPrecisions,
      },
      topPrecisions,
    )
    if (activeSpec) {
      models.unshift(activeSpec)
      seen.add(activeSpec.id)
    }
  }

  if (models.length === 0) return null

  const activeModel = rawActive && seen.has(rawActive) ? rawActive : models[0].id
  return {
    activeModel,
    models,
    device: typeof obj.device === 'string' ? obj.device : undefined,
    env: typeof obj.env === 'string' ? obj.env : undefined,
  }
}

export function resolveModelSpec(modelId: string, availableModels: VlaModelSpec[]): VlaModelSpec {
  const exact = availableModels.find((m) => m.id === modelId || m.label === modelId)
  if (exact) return exact
  return (
    buildModelSpecFromRaw(modelId) ?? {
      id: modelId || 'unknown-vla',
      label: modelId || 'Unknown VLA',
      family: 'VLA Policy',
      paramsLabel: 'N/A',
      baselineLatencyMs: 50,
      badge: 'LIVE',
    }
  )
}

export type ControlAction = 'reset' | 'pause' | 'resume' | 'export_gif'

export interface ControlMessage {
  type: 'control'
  action: ControlAction
}

export type ClientMessage = ChaosMessage | ConfigMessage | ControlMessage

export type AppView = 'arcade' | 'gallery'

export type ConnectionMode = 'live' | 'demo'
export type ConnectionState =
  | 'connecting'
  | 'connected'
  | 'reconnecting'
  | 'demo_fallback'
  | 'demo_manual'

export type EventCategory = 'chaos' | 'config' | 'control' | 'status' | 'system' | 'error'

export interface EventLogEntry {
  id: string
  timestamp: string
  category: EventCategory
  title: string
  detail: string
  severity?: 'info' | 'warn' | 'danger' | 'success'
}

export interface ChaosSkillMeta {
  skill: ChaosSkill
  labelZh: string
  labelEn: string
  category: 'physical' | 'visual' | 'systems'
  description: string
  cooldownMs: number
  hotkey?: string
  accent: 'rose' | 'orange' | 'cyan' | 'violet' | 'amber' | 'red'
}

export const CHAOS_SKILL_META: Record<ChaosSkill, ChaosSkillMeta> = {
  object_nudge: {
    skill: 'object_nudge',
    labelZh: '推一把',
    labelEn: 'OBJECT NUDGE',
    category: 'physical',
    description: 'Impulse target object position (qpos)',
    cooldownMs: 900,
    hotkey: '1',
    accent: 'rose',
  },
  arm_kick: {
    skill: 'arm_kick',
    labelZh: '撞机械臂',
    labelEn: 'ARM KICK',
    category: 'physical',
    description: 'External impulse on end-effector (xfrc)',
    cooldownMs: 1100,
    hotkey: '2',
    accent: 'orange',
  },
  friction_glitch: {
    skill: 'friction_glitch',
    labelZh: '桌面抹油',
    labelEn: 'FRICTION GLITCH',
    category: 'physical',
    description: 'Drop tabletop friction to 0.1x ice mode',
    cooldownMs: 2200,
    hotkey: '3',
    accent: 'cyan',
  },
  lighting_shift: {
    skill: 'lighting_shift',
    labelZh: '灯光突变',
    labelEn: 'LIGHTING SHIFT',
    category: 'visual',
    description: 'Sudden ambient brightness & chroma shift',
    cooldownMs: 1800,
    hotkey: '4',
    accent: 'violet',
  },
  camera_shake: {
    skill: 'camera_shake',
    labelZh: '镜头抖动',
    labelEn: 'CAMERA SHAKE',
    category: 'visual',
    description: 'High-frequency camera extrinsic jitter',
    cooldownMs: 1800,
    hotkey: '5',
    accent: 'violet',
  },
  occlusion_patch: {
    skill: 'occlusion_patch',
    labelZh: '遮挡贴纸',
    labelEn: 'OCCLUSION PATCH',
    category: 'visual',
    description: 'Slap opaque hazard patch on camera lens',
    cooldownMs: 2000,
    hotkey: '6',
    accent: 'amber',
  },
  latency_inject: {
    skill: 'latency_inject',
    labelZh: '延迟尖峰',
    labelEn: 'LATENCY SPIKE',
    category: 'systems',
    description: 'Inject temporary control-loop delay burst',
    cooldownMs: 1500,
    hotkey: '7',
    accent: 'amber',
  },
  wrist_blackout: {
    skill: 'wrist_blackout',
    labelZh: '腕部断流 2s',
    labelEn: 'WRIST BLACKOUT',
    category: 'systems',
    description: 'Drop wrist camera feed for 2 seconds',
    cooldownMs: 2600,
    hotkey: '8',
    accent: 'red',
  },
}

export interface PrecisionMeta {
  id: Precision
  label: string
  bits: string
  tag: string
  colorClass: string
}

export const PRECISION_OPTIONS: PrecisionMeta[] = [
  {
    id: 'bf16',
    label: 'BF16',
    bits: '16-bit',
    tag: 'REF',
    colorClass: 'text-emerald-300 border-emerald-400/60 bg-emerald-500/15',
  },
  {
    id: 'fp8',
    label: 'FP8',
    bits: '8-bit',
    tag: 'FAST',
    colorClass: 'text-cyan-300 border-cyan-400/60 bg-cyan-500/15',
  },
  {
    id: 'int8',
    label: 'INT8',
    bits: '8-bit',
    tag: 'QUANT',
    colorClass: 'text-amber-300 border-amber-400/60 bg-amber-500/15',
  },
  {
    id: 'nvfp4',
    label: 'NVFP4',
    bits: '4-bit',
    tag: 'TURBO',
    colorClass: 'text-fuchsia-300 border-fuchsia-400/60 bg-fuchsia-500/15',
  },
]

export const PRESET_PROMPTS: string[] = [
  'pick up the crimson block and place it in the cyan bowl',
  'open the top drawer and put the ceramic mug inside',
  'push the yellow mug to the right side of the stove',
  'stack the red block on top of the metallic plate',
]

export function toImageDataUri(raw: string): string {
  if (!raw) return ''
  if (raw.startsWith('data:image/')) return raw
  if (raw.startsWith('/9j/')) return `data:image/jpeg;base64,${raw}`
  if (raw.startsWith('iVBOR')) return `data:image/png;base64,${raw}`
  if (raw.startsWith('UklGR')) return `data:image/webp;base64,${raw}`
  if (raw.startsWith('R0lGOD')) return `data:image/gif;base64,${raw}`
  return `data:image/jpeg;base64,${raw}`
}

export function deriveHealthUrl(wsUrl: string): string | null {
  try {
    const parsed = new URL(wsUrl)
    const protocol = parsed.protocol === 'wss:' ? 'https:' : 'http:'
    return `${protocol}//${parsed.host}/health`
  } catch {
    return null
  }
}

export function formatTimestamp(date = new Date()): string {
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  const ss = String(date.getSeconds()).padStart(2, '0')
  const ms = String(date.getMilliseconds()).padStart(3, '0')
  return `${hh}:${mm}:${ss}.${ms}`
}
