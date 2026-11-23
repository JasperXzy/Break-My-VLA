import {
  type Action7D,
  type ActionChunk,
  type ChaosParamsMap,
  type ChaosSkill,
  type FrameMessage,
  type FrameStatus,
  type Precision,
  PRESET_PROMPTS,
} from '../types/protocol'

interface Vec3 {
  x: number
  y: number
  z: number
}

interface ActiveEffectState {
  remaining: number
  multiplier?: number
  brightness?: number
  saturation?: number
  amplitudePx?: number
  patch?: { x: number; y: number; width: number; height: number }
  latencyMs?: number
}

interface Spark {
  x: number
  y: number
  vx: number
  vy: number
  life: number
  color: string
}

const PRECISION_LATENCY_BASE: Record<Precision, number> = {
  bf16: 54,
  fp8: 31,
  int8: 27,
  nvfp4: 13,
}

const PRECISION_NOISE_SCALE: Record<Precision, number> = {
  bf16: 0.008,
  fp8: 0.018,
  int8: 0.032,
  nvfp4: 0.055,
}

const clamp = (v: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, v))

/**
 * Self-contained 2.5D tabletop manipulation simulator & dual-camera renderer
 * powering Local Demo Mode when no backend WebSocket is connected.
 */
export class DemoSimulationEngine {
  private step = 0
  private status: FrameStatus = 'running'
  private precision: Precision = 'bf16'
  private denoiseSteps = 8
  private baseLatencyMs = 0
  private prompt = PRESET_PROMPTS[0]

  // Normalized workspace coordinates: x in [-0.45, 0.45], y in [-0.32, 0.32], z in [0, 0.36]
  private eePos: Vec3 = { x: -0.18, y: 0.14, z: 0.24 }
  private eeVel: Vec3 = { x: 0, y: 0, z: 0 }
  private gripperOpen = 1.0
  private grasping = false

  private cubePos: Vec3 = { x: -0.22, y: -0.05, z: 0.03 }
  private cubeVel: Vec3 = { x: 0, y: 0, z: 0 }
  private goalPos: Vec3 = { x: 0.24, y: 0.04, z: 0.03 }

  private phase: 'approach' | 'descend' | 'grasp' | 'lift' | 'transport' | 'place' | 'done' =
    'approach'
  private holdCounter = 0
  private lastKickVector: Vec3 | null = null
  private kickFlashFrames = 0

  private effects = new Map<ChaosSkill, ActiveEffectState>()
  private delayedActionQueue: Action7D[] = []
  private sparks: Spark[] = []

  private mainCanvas: HTMLCanvasElement
  private mainCtx: CanvasRenderingContext2D
  private wristCanvas: HTMLCanvasElement
  private wristCtx: CanvasRenderingContext2D
  private gifCanvas: HTMLCanvasElement
  private gifCtx: CanvasRenderingContext2D
  private gifFrames: ImageData[] = []

  constructor() {
    this.mainCanvas = document.createElement('canvas')
    this.mainCanvas.width = 480
    this.mainCanvas.height = 300
    const mainCtx = this.mainCanvas.getContext('2d')
    if (!mainCtx) throw new Error('2D context unavailable')
    this.mainCtx = mainCtx

    this.wristCanvas = document.createElement('canvas')
    this.wristCanvas.width = 320
    this.wristCanvas.height = 200
    const wristCtx = this.wristCanvas.getContext('2d')
    if (!wristCtx) throw new Error('2D context unavailable')
    this.wristCtx = wristCtx

    this.gifCanvas = document.createElement('canvas')
    this.gifCanvas.width = 240
    this.gifCanvas.height = 150
    const gifCtx = this.gifCanvas.getContext('2d', { willReadFrequently: true })
    if (!gifCtx) throw new Error('2D context unavailable')
    this.gifCtx = gifCtx
  }

  public reset(): void {
    this.step = 0
    this.status = 'running'
    this.eePos = { x: -0.18, y: 0.14, z: 0.24 }
    this.eeVel = { x: 0, y: 0, z: 0 }
    this.gripperOpen = 1.0
    this.grasping = false
    this.cubePos = {
      x: -0.22 + (Math.random() - 0.5) * 0.06,
      y: -0.06 + (Math.random() - 0.5) * 0.06,
      z: 0.03,
    }
    this.cubeVel = { x: 0, y: 0, z: 0 }
    this.phase = 'approach'
    this.holdCounter = 0
    this.effects.clear()
    this.delayedActionQueue = []
    this.sparks = []
    this.kickFlashFrames = 0
    this.gifFrames = []
  }

  public setPaused(paused: boolean): void {
    if (this.status === 'success') return
    this.status = paused ? 'paused' : 'running'
  }

  public configure(cfg: {
    precision?: Precision
    denoise_steps?: number
    latency_ms?: number
    prompt?: string
  }): void {
    if (cfg.precision) this.precision = cfg.precision
    if (typeof cfg.denoise_steps === 'number') {
      this.denoiseSteps = clamp(Math.round(cfg.denoise_steps), 1, 10)
    }
    if (typeof cfg.latency_ms === 'number') {
      this.baseLatencyMs = clamp(cfg.latency_ms, 0, 300)
    }
    if (typeof cfg.prompt === 'string') {
      this.prompt = cfg.prompt
    }
  }

  public injectChaos<K extends ChaosSkill>(skill: K, params: ChaosParamsMap[K]): void {
    switch (skill) {
      case 'object_nudge': {
        const p = params as ChaosParamsMap['object_nudge']
        this.grasping = false
        this.gripperOpen = 1.0
        this.cubePos.x = clamp(this.cubePos.x + p.dx * 0.65, -0.38, 0.38)
        this.cubePos.y = clamp(this.cubePos.y + p.dy * 0.65, -0.24, 0.24)
        this.cubeVel.x += p.dx * 0.14
        this.cubeVel.y += p.dy * 0.14
        this.cubeVel.z += Math.abs(p.dz) * 0.08 + 0.025
        this.phase = 'approach'
        if (this.status === 'success') this.status = 'running'
        this.effects.set('object_nudge', { remaining: 14 })
        this.spawnSparks(this.cubePos, '#fb7185')
        break
      }
      case 'arm_kick': {
        const p = params as ChaosParamsMap['arm_kick']
        this.eeVel.x += p.dx * 0.24
        this.eeVel.y += p.dy * 0.24
        this.eeVel.z += p.dz * 0.2
        this.lastKickVector = { x: p.dx, y: p.dy, z: p.dz }
        this.kickFlashFrames = 14
        if (this.grasping && Math.random() < 0.72) {
          this.grasping = false
          this.cubeVel.x += p.dx * 0.12
          this.cubeVel.y += p.dy * 0.12
          this.phase = 'approach'
        }
        this.effects.set('arm_kick', { remaining: 16 })
        this.spawnSparks(this.eePos, '#fb923c')
        break
      }
      case 'friction_glitch': {
        const p = params as ChaosParamsMap['friction_glitch']
        this.effects.set('friction_glitch', {
          remaining: Math.max(12, p.duration_steps),
          multiplier: p.multiplier,
        })
        if (!this.grasping) {
          this.cubeVel.x += (Math.random() - 0.5) * 0.04
          this.cubeVel.y += (Math.random() - 0.5) * 0.04
        }
        break
      }
      case 'lighting_shift': {
        const p = params as ChaosParamsMap['lighting_shift']
        this.effects.set('lighting_shift', {
          remaining: Math.max(16, p.duration_steps),
          brightness: p.brightness,
          saturation: p.saturation,
        })
        break
      }
      case 'camera_shake': {
        const p = params as ChaosParamsMap['camera_shake']
        this.effects.set('camera_shake', {
          remaining: Math.max(16, p.duration_steps),
          amplitudePx: p.amplitude_px,
        })
        break
      }
      case 'occlusion_patch': {
        const p = params as ChaosParamsMap['occlusion_patch']
        this.effects.set('occlusion_patch', {
          remaining: Math.max(20, p.duration_steps),
          patch: { x: p.x, y: p.y, width: p.width, height: p.height },
        })
        break
      }
      case 'latency_inject': {
        const p = params as ChaosParamsMap['latency_inject']
        this.effects.set('latency_inject', {
          remaining: Math.max(18, p.duration_steps),
          latencyMs: p.latency_ms,
        })
        break
      }
      case 'wrist_blackout': {
        const p = params as ChaosParamsMap['wrist_blackout']
        this.effects.set('wrist_blackout', {
          remaining: Math.max(20, p.duration_steps),
        })
        break
      }
    }
  }

  public getRecordedGifFrames(): ImageData[] {
    return this.gifFrames
  }

  private spawnSparks(pos: Vec3, color: string): void {
    const pt = this.project(pos.x, pos.y, pos.z)
    for (let i = 0; i < 14; i++) {
      const ang = (Math.PI * 2 * i) / 14 + Math.random() * 0.4
      const speed = 1.6 + Math.random() * 3.8
      this.sparks.push({
        x: pt.x,
        y: pt.y,
        vx: Math.cos(ang) * speed,
        vy: Math.sin(ang) * speed - 1.2,
        life: 1.0,
        color,
      })
    }
  }

  private computeTarget(): { target: Vec3; desiredGripper: number } {
    const dCubeXY = Math.hypot(this.eePos.x - this.cubePos.x, this.eePos.y - this.cubePos.y)
    const dGoalXY = Math.hypot(this.cubePos.x - this.goalPos.x, this.cubePos.y - this.goalPos.y)

    if (this.phase === 'done') {
      return {
        target: { x: this.goalPos.x, y: this.goalPos.y, z: 0.24 },
        desiredGripper: 1.0,
      }
    }

    if (!this.grasping) {
      if (this.phase === 'approach') {
        if (dCubeXY < 0.032 && Math.abs(this.eePos.z - 0.14) < 0.04) {
          this.phase = 'descend'
        }
        return {
          target: { x: this.cubePos.x, y: this.cubePos.y, z: 0.14 },
          desiredGripper: 1.0,
        }
      }

      if (this.phase === 'descend') {
        if (dCubeXY > 0.075) {
          this.phase = 'approach'
        } else if (Math.abs(this.eePos.z - (this.cubePos.z + 0.02)) < 0.025) {
          this.phase = 'grasp'
          this.holdCounter = 0
        }
        return {
          target: { x: this.cubePos.x, y: this.cubePos.y, z: this.cubePos.z + 0.02 },
          desiredGripper: 1.0,
        }
      }

      if (this.phase === 'grasp') {
        this.holdCounter++
        if (dCubeXY > 0.065) {
          this.phase = 'approach'
        } else if (this.holdCounter > 6) {
          this.grasping = true
          this.phase = 'lift'
        }
        return {
          target: { x: this.cubePos.x, y: this.cubePos.y, z: this.cubePos.z + 0.015 },
          desiredGripper: 0.0,
        }
      }
    }

    if (this.phase === 'lift') {
      if (this.eePos.z > 0.19) {
        this.phase = 'transport'
      }
      return {
        target: { x: this.cubePos.x, y: this.cubePos.y, z: 0.22 },
        desiredGripper: 0.0,
      }
    }

    if (this.phase === 'transport') {
      if (dGoalXY < 0.042) {
        this.phase = 'place'
      }
      return {
        target: { x: this.goalPos.x, y: this.goalPos.y, z: 0.21 },
        desiredGripper: 0.0,
      }
    }

    if (this.phase === 'place') {
      if (this.eePos.z < 0.075) {
        this.grasping = false
        this.cubePos.x = this.goalPos.x
        this.cubePos.y = this.goalPos.y
        this.cubePos.z = 0.03
        this.phase = 'done'
        this.holdCounter = 0
      }
      return {
        target: { x: this.goalPos.x, y: this.goalPos.y, z: 0.055 },
        desiredGripper: 0.0,
      }
    }

    return {
      target: { x: this.cubePos.x, y: this.cubePos.y, z: 0.16 },
      desiredGripper: 1.0,
    }
  }

  public tick(): FrameMessage {
    const activeChaosList: ChaosSkill[] = []
    for (const [skill, state] of this.effects.entries()) {
      if (state.remaining > 0) {
        activeChaosList.push(skill)
      }
    }

    const latencySpike = this.effects.get('latency_inject')?.latencyMs ?? 0
    const effectiveInjectedLatency = Math.max(this.baseLatencyMs, latencySpike)
    if (this.baseLatencyMs > 0 && !activeChaosList.includes('latency_inject')) {
      activeChaosList.push('latency_inject')
    }

    const baseInfer =
      PRECISION_LATENCY_BASE[this.precision] +
      this.denoiseSteps * (this.precision === 'nvfp4' ? 1.3 : this.precision === 'bf16' ? 3.8 : 2.2)
    const inferJitter = (Math.random() - 0.5) * 4.5
    const inferMs = Number(Math.max(4, baseInfer + effectiveInjectedLatency + inferJitter).toFixed(1))

    // Noise increases with lower precision, fewer denoise steps, visual chaos, and wrist blackout
    const denoisePenalty = (11 - this.denoiseSteps) * 0.0042
    const visualDisruption =
      (this.effects.has('occlusion_patch') ? 0.042 : 0) +
      (this.effects.has('camera_shake') ? 0.034 : 0) +
      (this.effects.has('lighting_shift') ? 0.025 : 0) +
      (this.effects.has('wrist_blackout') ? 0.038 : 0)
    const totalNoise = PRECISION_NOISE_SCALE[this.precision] + denoisePenalty + visualDisruption

    const { target, desiredGripper } = this.computeTarget()

    // Synthesize 16-step predicted action chunk
    const actionChunk: ActionChunk = []
    let simX = this.eePos.x
    let simY = this.eePos.y
    let simZ = this.eePos.z
    for (let k = 0; k < 16; k++) {
      const dx = clamp((target.x - simX) * 0.24, -0.14, 0.14)
      const dy = clamp((target.y - simY) * 0.24, -0.14, 0.14)
      const dz = clamp((target.z - simZ) * 0.24, -0.14, 0.14)
      const wave = Math.sin(this.step * 0.25 + k * 0.55) * totalNoise
      const ndx = Number((dx + (Math.random() - 0.5) * totalNoise + wave * 0.5).toFixed(4))
      const ndy = Number((dy + (Math.random() - 0.5) * totalNoise - wave * 0.4).toFixed(4))
      const ndz = Number((dz + (Math.random() - 0.5) * totalNoise * 0.6).toFixed(4))
      const rx = Number((Math.sin(this.step * 0.12 + k * 0.3) * 0.12 + wave).toFixed(4))
      const ry = Number((Math.cos(this.step * 0.1 + k * 0.25) * 0.1 - wave * 0.6).toFixed(4))
      const rz = Number(((target.x - simX) * 0.35).toFixed(4))
      const grip = Number(clamp(desiredGripper + (Math.random() - 0.5) * totalNoise, 0, 1).toFixed(3))
      actionChunk.push([ndx, ndy, ndz, rx, ry, rz, grip])
      simX += ndx * 0.4
      simY += ndy * 0.4
      simZ += ndz * 0.4
    }

    if (this.status === 'running') {
      this.step++

      // Queue action with delay proportional to latency
      const delayFrames = Math.round(effectiveInjectedLatency / 45)
      this.delayedActionQueue.push(actionChunk[0])
      while (this.delayedActionQueue.length > delayFrames + 1) {
        this.delayedActionQueue.shift()
      }
      const applied = this.delayedActionQueue[0] ?? actionChunk[0]

      // Apply PD + chaos dynamics
      const trackingGain = this.effects.has('wrist_blackout') ? 0.52 : 0.78
      this.eeVel.x = this.eeVel.x * 0.62 + applied[0] * 0.34 * trackingGain
      this.eeVel.y = this.eeVel.y * 0.62 + applied[1] * 0.34 * trackingGain
      this.eeVel.z = this.eeVel.z * 0.62 + applied[2] * 0.34 * trackingGain

      this.eePos.x = clamp(this.eePos.x + this.eeVel.x, -0.44, 0.44)
      this.eePos.y = clamp(this.eePos.y + this.eeVel.y, -0.3, 0.3)
      this.eePos.z = clamp(this.eePos.z + this.eeVel.z, 0.04, 0.35)
      this.gripperOpen = clamp(this.gripperOpen * 0.72 + applied[6] * 0.28, 0, 1)

      if (this.grasping) {
        this.cubePos.x = this.eePos.x
        this.cubePos.y = this.eePos.y
        this.cubePos.z = Math.max(0.03, this.eePos.z - 0.02)
        this.cubeVel = { x: 0, y: 0, z: 0 }
      } else {
        const frictionEffect = this.effects.get('friction_glitch')
        const damping = frictionEffect ? 0.985 : 0.78
        this.cubePos.x += this.cubeVel.x
        this.cubePos.y += this.cubeVel.y
        this.cubePos.z += this.cubeVel.z
        this.cubeVel.x *= damping
        this.cubeVel.y *= damping
        this.cubeVel.z -= 0.012
        if (this.cubePos.z <= 0.03) {
          this.cubePos.z = 0.03
          this.cubeVel.z = Math.abs(this.cubeVel.z) > 0.018 ? -this.cubeVel.z * 0.35 : 0
        }
        // Bounce softly off table edges
        if (this.cubePos.x < -0.4 || this.cubePos.x > 0.4) {
          this.cubePos.x = clamp(this.cubePos.x, -0.4, 0.4)
          this.cubeVel.x *= -0.7
        }
        if (this.cubePos.y < -0.26 || this.cubePos.y > 0.26) {
          this.cubePos.y = clamp(this.cubePos.y, -0.26, 0.26)
          this.cubeVel.y *= -0.7
        }
      }

      // Decrement timed effects
      for (const [skill, state] of this.effects.entries()) {
        state.remaining--
        if (state.remaining <= 0) {
          this.effects.delete(skill)
        }
      }
      if (this.kickFlashFrames > 0) this.kickFlashFrames--

      // Update sparks
      this.sparks = this.sparks
        .map((s) => ({
          ...s,
          x: s.x + s.vx,
          y: s.y + s.vy,
          vy: s.vy + 0.16,
          life: s.life - 0.065,
        }))
        .filter((s) => s.life > 0)

      // Check completion
      const distToGoal = Math.hypot(
        this.cubePos.x - this.goalPos.x,
        this.cubePos.y - this.goalPos.y,
      )
      if (this.phase === 'done' && distToGoal < 0.065) {
        this.holdCounter++
        if (this.holdCounter > 16) {
          this.status = 'success'
        }
      }
    }

    // Compute progress 0..1
    const distStart = 0.48
    const distNow = Math.hypot(this.cubePos.x - this.goalPos.x, this.cubePos.y - this.goalPos.y)
    let progress = clamp(1 - distNow / distStart, 0, 1) * 0.72
    if (this.grasping) progress = Math.max(progress, 0.35)
    if (this.phase === 'lift') progress = Math.max(progress, 0.44)
    if (this.phase === 'done' || this.status === 'success') progress = 1.0
    progress = Number(clamp(progress, 0, 1).toFixed(3))

    this.renderMainView(activeChaosList)
    this.renderWristView(activeChaosList)

    // Record frame every 2 steps for Demo GIF export (rolling ~3.5s)
    if (this.step % 2 === 0) {
      this.gifCtx.drawImage(this.mainCanvas, 0, 0, this.gifCanvas.width, this.gifCanvas.height)
      const imgData = this.gifCtx.getImageData(0, 0, this.gifCanvas.width, this.gifCanvas.height)
      this.gifFrames.push(imgData)
      if (this.gifFrames.length > 36) {
        this.gifFrames.shift()
      }
    }

    const fps = Number(clamp(1000 / Math.max(48, inferMs * 0.85), 4.5, 24.0).toFixed(1))

    return {
      type: 'frame',
      step: this.step,
      fps,
      infer_ms: inferMs,
      precision: this.precision,
      denoise_steps: this.denoiseSteps,
      rgb_main_b64: this.mainCanvas.toDataURL('image/jpeg', 0.82),
      rgb_wrist_b64: this.wristCanvas.toDataURL('image/jpeg', 0.8),
      action_chunk: actionChunk,
      active_chaos: activeChaosList,
      task_progress: progress,
      status: this.status,
      model: 'demo-vla-sim (local)',
    }
  }

  private project(x: number, y: number, z: number): { x: number; y: number } {
    const cx = 240
    const cy = 204
    const sx = cx + x * 420 - y * 105
    const sy = cy + y * 185 - z * 345
    return { x: sx, y: sy }
  }

  private renderMainView(activeChaos: ChaosSkill[]): void {
    const ctx = this.mainCtx
    const w = this.mainCanvas.width
    const h = this.mainCanvas.height

    ctx.save()
    ctx.clearRect(0, 0, w, h)

    // Camera shake offset
    const shake = this.effects.get('camera_shake')
    if (shake) {
      const amp = (shake.amplitudePx ?? 14) * 0.65
      ctx.translate((Math.random() - 0.5) * amp, (Math.random() - 0.5) * amp)
    }

    // Lab room backdrop
    const bgGrad = ctx.createLinearGradient(0, 0, 0, h)
    bgGrad.addColorStop(0, '#091124')
    bgGrad.addColorStop(0.55, '#0f1b33')
    bgGrad.addColorStop(1, '#080e1c')
    ctx.fillStyle = bgGrad
    ctx.fillRect(-30, -30, w + 60, h + 60)

    // Subtle background horizon grid
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.08)'
    ctx.lineWidth = 1
    for (let x = 0; x <= w; x += 32) {
      ctx.beginPath()
      ctx.moveTo(x, 0)
      ctx.lineTo(x, h)
      ctx.stroke()
    }

    // Tabletop quad
    const tFL = this.project(-0.46, 0.31, 0)
    const tFR = this.project(0.46, 0.31, 0)
    const tBR = this.project(0.46, -0.31, 0)
    const tBL = this.project(-0.46, -0.31, 0)

    const isIcy = activeChaos.includes('friction_glitch')
    const tableGrad = ctx.createLinearGradient(0, 120, 0, 290)
    if (isIcy) {
      tableGrad.addColorStop(0, '#164e63')
      tableGrad.addColorStop(1, '#083344')
    } else {
      tableGrad.addColorStop(0, '#1e293b')
      tableGrad.addColorStop(1, '#0f172a')
    }

    ctx.beginPath()
    ctx.moveTo(tFL.x, tFL.y)
    ctx.lineTo(tFR.x, tFR.y)
    ctx.lineTo(tBR.x, tBR.y)
    ctx.lineTo(tBL.x, tBL.y)
    ctx.closePath()
    ctx.fillStyle = tableGrad
    ctx.fill()
    ctx.strokeStyle = isIcy ? '#22d3ee' : '#334155'
    ctx.lineWidth = isIcy ? 2 : 1.5
    ctx.stroke()

    // Tabletop coordinate grid
    ctx.strokeStyle = isIcy ? 'rgba(103, 232, 249, 0.25)' : 'rgba(148, 163, 184, 0.14)'
    ctx.lineWidth = 1
    for (let gx = -0.4; gx <= 0.41; gx += 0.1) {
      const p1 = this.project(gx, -0.3, 0)
      const p2 = this.project(gx, 0.3, 0)
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }
    for (let gy = -0.25; gy <= 0.26; gy += 0.1) {
      const p1 = this.project(-0.45, gy, 0)
      const p2 = this.project(0.45, gy, 0)
      ctx.beginPath()
      ctx.moveTo(p1.x, p1.y)
      ctx.lineTo(p2.x, p2.y)
      ctx.stroke()
    }

    // Target bowl / goal zone
    const goalPt = this.project(this.goalPos.x, this.goalPos.y, 0.005)
    ctx.save()
    ctx.beginPath()
    ctx.ellipse(goalPt.x, goalPt.y, 38, 17, 0, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(6, 182, 212, 0.16)'
    ctx.fill()
    ctx.strokeStyle = '#22d3ee'
    ctx.lineWidth = 2
    ctx.stroke()
    ctx.beginPath()
    ctx.ellipse(goalPt.x, goalPt.y - 4, 30, 12, 0, 0, Math.PI * 2)
    ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)'
    ctx.lineWidth = 1.2
    ctx.stroke()
    ctx.fillStyle = '#67e8f9'
    ctx.font = '600 9px JetBrains Mono, monospace'
    ctx.textAlign = 'center'
    ctx.fillText('GOAL ZONE', goalPt.x, goalPt.y + 27)
    ctx.restore()

    // Cube shadow
    const cubeShadow = this.project(this.cubePos.x, this.cubePos.y, 0.002)
    ctx.beginPath()
    ctx.ellipse(cubeShadow.x, cubeShadow.y, 16, 7, 0, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(0, 0, 0, 0.42)'
    ctx.fill()

    // Graspable crimson cube
    const cPt = this.project(this.cubePos.x, this.cubePos.y, this.cubePos.z)
    const s = 14
    // Front face
    ctx.fillStyle = '#e11d48'
    ctx.fillRect(cPt.x - s, cPt.y - s, s * 2, s * 1.65)
    // Top face
    ctx.beginPath()
    ctx.moveTo(cPt.x - s, cPt.y - s)
    ctx.lineTo(cPt.x - s + 8, cPt.y - s - 8)
    ctx.lineTo(cPt.x + s + 8, cPt.y - s - 8)
    ctx.lineTo(cPt.x + s, cPt.y - s)
    ctx.closePath()
    ctx.fillStyle = '#fb7185'
    ctx.fill()
    // Right face
    ctx.beginPath()
    ctx.moveTo(cPt.x + s, cPt.y - s)
    ctx.lineTo(cPt.x + s + 8, cPt.y - s - 8)
    ctx.lineTo(cPt.x + s + 8, cPt.y + s * 0.65 - 8)
    ctx.lineTo(cPt.x + s, cPt.y + s * 0.65)
    ctx.closePath()
    ctx.fillStyle = '#9f1239'
    ctx.fill()
    ctx.strokeStyle = 'rgba(255, 228, 230, 0.45)'
    ctx.lineWidth = 1
    ctx.strokeRect(cPt.x - s, cPt.y - s, s * 2, s * 1.65)

    // Articulated robot arm (Franka-style kinematic chain)
    const base = this.project(-0.02, -0.33, 0.02)
    const shoulder = this.project(-0.02, -0.33, 0.24)
    const elbow = this.project(
      (this.eePos.x - 0.02) * 0.48,
      (this.eePos.y - 0.33) * 0.55,
      Math.max(0.28, this.eePos.z + 0.14),
    )
    const wrist = this.project(this.eePos.x, this.eePos.y, this.eePos.z + 0.05)
    const ee = this.project(this.eePos.x, this.eePos.y, this.eePos.z)

    // Pedestal
    ctx.fillStyle = '#334155'
    ctx.beginPath()
    ctx.ellipse(base.x, base.y, 24, 10, 0, 0, Math.PI * 2)
    ctx.fill()
    ctx.strokeStyle = '#64748b'
    ctx.lineWidth = 1.5
    ctx.stroke()

    const drawLink = (
      a: { x: number; y: number },
      b: { x: number; y: number },
      width: number,
      color: string,
    ) => {
      ctx.strokeStyle = '#020617'
      ctx.lineWidth = width + 3
      ctx.lineCap = 'round'
      ctx.beginPath()
      ctx.moveTo(a.x, a.y)
      ctx.lineTo(b.x, b.y)
      ctx.stroke()

      ctx.strokeStyle = color
      ctx.lineWidth = width
      ctx.beginPath()
      ctx.moveTo(a.x, a.y)
      ctx.lineTo(b.x, b.y)
      ctx.stroke()
    }

    drawLink(base, shoulder, 14, '#cbd5e1')
    drawLink(shoulder, elbow, 11, '#e2e8f0')
    drawLink(elbow, wrist, 9, '#f1f5f9')
    drawLink(wrist, ee, 6, '#38bdf8')

    // Joints
    for (const j of [shoulder, elbow, wrist]) {
      ctx.beginPath()
      ctx.arc(j.x, j.y, 6.5, 0, Math.PI * 2)
      ctx.fillStyle = '#0f172a'
      ctx.fill()
      ctx.strokeStyle = '#38bdf8'
      ctx.lineWidth = 2
      ctx.stroke()
    }

    // Parallel-jaw gripper
    const jawSpread = 5 + this.gripperOpen * 10
    ctx.strokeStyle = this.grasping ? '#34d399' : '#e2e8f0'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.moveTo(ee.x - jawSpread, ee.y - 2)
    ctx.lineTo(ee.x - jawSpread, ee.y + 13)
    ctx.moveTo(ee.x + jawSpread, ee.y - 2)
    ctx.lineTo(ee.x + jawSpread, ee.y + 13)
    ctx.moveTo(ee.x - jawSpread, ee.y - 2)
    ctx.lineTo(ee.x + jawSpread, ee.y - 2)
    ctx.stroke()

    // Arm kick impact arrow
    if (this.kickFlashFrames > 0 && this.lastKickVector) {
      ctx.save()
      ctx.strokeStyle = '#fb923c'
      ctx.lineWidth = 3
      ctx.beginPath()
      ctx.moveTo(ee.x - this.lastKickVector.x * 160, ee.y - this.lastKickVector.y * 120)
      ctx.lineTo(ee.x, ee.y)
      ctx.stroke()
      ctx.fillStyle = '#fdba74'
      ctx.font = '700 11px JetBrains Mono, monospace'
      ctx.fillText('IMPULSE!', ee.x + 14, ee.y - 12)
      ctx.restore()
    }

    // Sparks
    for (const sp of this.sparks) {
      ctx.fillStyle = sp.color
      ctx.globalAlpha = sp.life
      ctx.fillRect(sp.x, sp.y, 3.5, 3.5)
    }
    ctx.globalAlpha = 1

    // Lighting shift overlay
    const light = this.effects.get('lighting_shift')
    if (light) {
      const dim = clamp(1 - (light.brightness ?? 0.45), 0, 0.75)
      const strobe = 0.08 * Math.sin(this.step * 0.9)
      ctx.fillStyle = `rgba(38, 12, 76, ${clamp(dim + strobe, 0.15, 0.75)})`
      ctx.fillRect(-30, -30, w + 60, h + 60)
    }

    // Occlusion hazard patch
    const occ = this.effects.get('occlusion_patch')
    if (occ?.patch) {
      const px = occ.patch.x * w
      const py = occ.patch.y * h
      const pw = occ.patch.width * w
      const ph = occ.patch.height * h
      ctx.save()
      ctx.fillStyle = '#111827'
      ctx.fillRect(px, py, pw, ph)
      ctx.strokeStyle = '#facc15'
      ctx.lineWidth = 2.5
      ctx.strokeRect(px, py, pw, ph)
      ctx.beginPath()
      ctx.rect(px, py, pw, ph)
      ctx.clip()
      ctx.strokeStyle = 'rgba(250, 204, 21, 0.35)'
      ctx.lineWidth = 10
      for (let d = -ph; d < pw + ph; d += 24) {
        ctx.beginPath()
        ctx.moveTo(px + d, py)
        ctx.lineTo(px + d - ph, py + ph)
        ctx.stroke()
      }
      ctx.fillStyle = '#fef08a'
      ctx.font = '700 11px JetBrains Mono, monospace'
      ctx.textAlign = 'center'
      ctx.fillText('OCCLUSION', px + pw / 2, py + ph / 2 + 4)
      ctx.restore()
    }

    // Sub-prompt caption in frame bottom-left
    ctx.fillStyle = 'rgba(2, 6, 23, 0.65)'
    ctx.fillRect(8, h - 24, w - 16, 18)
    ctx.fillStyle = '#94a3b8'
    ctx.font = '500 10px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`PROMPT: ${this.prompt.slice(0, 56)}`, 14, h - 12)

    ctx.restore()
  }

  private renderWristView(activeChaos: ChaosSkill[]): void {
    const ctx = this.wristCtx
    const w = this.wristCanvas.width
    const h = this.wristCanvas.height

    ctx.save()
    ctx.clearRect(0, 0, w, h)

    if (activeChaos.includes('wrist_blackout')) {
      ctx.fillStyle = '#020617'
      ctx.fillRect(0, 0, w, h)
      ctx.restore()
      return
    }

    const isIcy = activeChaos.includes('friction_glitch')
    ctx.fillStyle = isIcy ? '#0c4a6e' : '#0f172a'
    ctx.fillRect(0, 0, w, h)

    // Relative position of cube & goal from end-effector perspective
    const relCubeX = (this.cubePos.x - this.eePos.x) * 480
    const relCubeY = (this.cubePos.y - this.eePos.y) * 480
    const scale = clamp(1.35 - this.eePos.z * 2.1, 0.55, 1.45)

    // Table grid relative to wrist
    ctx.strokeStyle = isIcy ? 'rgba(103, 232, 249, 0.25)' : 'rgba(148, 163, 184, 0.14)'
    ctx.lineWidth = 1
    const ox = ((-this.eePos.x * 480) % 36 + 36) % 36
    const oy = ((-this.eePos.y * 480) % 36 + 36) % 36
    for (let gx = ox; gx < w; gx += 36) {
      ctx.beginPath()
      ctx.moveTo(gx, 0)
      ctx.lineTo(gx, h)
      ctx.stroke()
    }
    for (let gy = oy; gy < h; gy += 36) {
      ctx.beginPath()
      ctx.moveTo(0, gy)
      ctx.lineTo(w, gy)
      ctx.stroke()
    }

    // Goal bowl in wrist frame
    const relGoalX = w / 2 + (this.goalPos.x - this.eePos.x) * 480
    const relGoalY = h / 2 + (this.goalPos.y - this.eePos.y) * 480
    ctx.beginPath()
    ctx.arc(relGoalX, relGoalY, 44 * scale, 0, Math.PI * 2)
    ctx.fillStyle = 'rgba(34, 211, 238, 0.14)'
    ctx.fill()
    ctx.strokeStyle = '#22d3ee'
    ctx.lineWidth = 2
    ctx.stroke()

    // Cube in wrist frame
    const cx = w / 2 + relCubeX
    const cy = h / 2 + relCubeY
    const size = 34 * scale
    ctx.fillStyle = '#e11d48'
    ctx.fillRect(cx - size / 2, cy - size / 2, size, size)
    ctx.strokeStyle = '#fda4af'
    ctx.lineWidth = 2
    ctx.strokeRect(cx - size / 2, cy - size / 2, size, size)

    // Gripper fingers framing bottom edges
    const fingerOffset = 22 + this.gripperOpen * 54
    ctx.fillStyle = '#334155'
    ctx.fillRect(w / 2 - fingerOffset - 22, h - 42, 22, 42)
    ctx.fillRect(w / 2 + fingerOffset, h - 42, 22, 42)
    ctx.strokeStyle = this.grasping ? '#34d399' : '#38bdf8'
    ctx.lineWidth = 2
    ctx.strokeRect(w / 2 - fingerOffset - 22, h - 42, 22, 42)
    ctx.strokeRect(w / 2 + fingerOffset, h - 42, 22, 42)

    // Crosshair reticle
    ctx.strokeStyle = 'rgba(56, 189, 248, 0.45)'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(w / 2 - 14, h / 2)
    ctx.lineTo(w / 2 + 14, h / 2)
    ctx.moveTo(w / 2, h / 2 - 14)
    ctx.lineTo(w / 2, h / 2 + 14)
    ctx.stroke()

    // Lighting shift tint on wrist cam too
    const light = this.effects.get('lighting_shift')
    if (light) {
      ctx.fillStyle = 'rgba(46, 16, 101, 0.45)'
      ctx.fillRect(0, 0, w, h)
    }

    ctx.restore()
  }
}
