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
  precision?: Precision
  denoise_steps?: number
  latency_ms?: number
  prompt?: string
}

export type ControlAction = 'reset' | 'pause' | 'resume' | 'export_gif'

export interface ControlMessage {
  type: 'control'
  action: ControlAction
}

export type ClientMessage = ChaosMessage | ConfigMessage | ControlMessage

export type ConnectionMode = 'live' | 'demo'
export type ConnectionState = 'connecting' | 'connected' | 'demo_fallback' | 'demo_manual'

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
