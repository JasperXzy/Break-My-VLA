import { useEffect, useState } from 'react'
import {
  Brain,
  Camera,
  Clock,
  Droplets,
  EyeOff,
  Hand,
  Move,
  Sparkles,
  SunDim,
  Vibrate,
  VideoOff,
  Zap,
} from 'lucide-react'
import {
  CHAOS_SKILL_META,
  type ChaosSkill,
  PRECISION_OPTIONS,
  type Precision,
} from '../types/protocol'

interface ChaosConsoleProps {
  activeChaos: ChaosSkill[]
  cooldowns: Partial<Record<ChaosSkill, { startedAt: number; endsAt: number }>>
  latencyMs: number
  precision: Precision
  denoiseSteps: number
  onTriggerChaos: (skill: ChaosSkill) => void
  onChangeLatency: (ms: number, emitLog?: boolean) => void
  onChangePrecision: (precision: Precision) => void
  onChangeDenoiseSteps: (steps: number) => void
}

interface ChaosButtonProps {
  skill: ChaosSkill
  icon: React.ReactNode
  isActive: boolean
  cooldownPct: number
  onClick: () => void
  accentClass: string
}

function ChaosSkillButton({
  skill,
  icon,
  isActive,
  cooldownPct,
  onClick,
  accentClass,
}: ChaosButtonProps) {
  const meta = CHAOS_SKILL_META[skill]
  const onCooldown = cooldownPct > 0

  return (
    <button
      type="button"
      onClick={onClick}
      className={`group relative flex w-full items-center justify-between overflow-hidden rounded-lg border px-2.5 py-2 text-left transition-all active:scale-[0.97] ${
        isActive
          ? 'border-rose-400/80 bg-rose-500/20 shadow-[0_0_16px_rgba(244,63,94,0.3)]'
          : accentClass
      }`}
    >
      {/* Cooldown sweep progress bar */}
      {onCooldown && (
        <span
          className="pointer-events-none absolute inset-y-0 left-0 bg-white/10 transition-[width] duration-75"
          style={{ width: `${cooldownPct}%` }}
        />
      )}

      <div className="relative z-10 flex items-center gap-2.5">
        <div
          className={`flex h-7 w-7 shrink-0 items-center justify-center rounded-md border ${
            isActive
              ? 'border-rose-400/60 bg-rose-500/25 text-rose-200'
              : 'border-slate-700/80 bg-slate-950/75 text-slate-200 group-hover:border-white/30'
          }`}
        >
          {icon}
        </div>
        <div className="leading-tight">
          <div className="flex items-center gap-1.5">
            <span className="text-xs font-bold text-white">{meta.labelZh}</span>
            {isActive && (
              <span className="h-1.5 w-1.5 animate-ping rounded-full bg-rose-400" />
            )}
          </div>
          <div className="font-mono text-[10px] text-slate-400">{meta.skill}</div>
        </div>
      </div>

      <div className="relative z-10 flex items-center gap-1.5 font-mono text-[10px]">
        {meta.hotkey && (
          <kbd className="rounded border border-slate-700 bg-slate-950/90 px-1.5 py-0.5 text-[9px] text-slate-400 group-hover:border-cyan-500/40 group-hover:text-cyan-300">
            {meta.hotkey}
          </kbd>
        )}
      </div>
    </button>
  )
}

export function ChaosConsole({
  activeChaos,
  cooldowns,
  latencyMs,
  precision,
  denoiseSteps,
  onTriggerChaos,
  onChangeLatency,
  onChangePrecision,
  onChangeDenoiseSteps,
}: ChaosConsoleProps) {
  const [now, setNow] = useState(() => Date.now())

  const hasActiveCooldown = Object.values(cooldowns).some(
    (entry) => entry && entry.endsAt > now,
  )

  useEffect(() => {
    if (!hasActiveCooldown) return
    const timer = window.setInterval(() => setNow(Date.now()), 50)
    return () => window.clearInterval(timer)
  }, [hasActiveCooldown])

  const getCooldownPct = (skill: ChaosSkill): number => {
    const cd = cooldowns[skill]
    if (!cd || cd.endsAt <= now) return 0
    const total = Math.max(1, cd.endsAt - cd.startedAt)
    return Math.min(100, Math.max(0, ((cd.endsAt - now) / total) * 100))
  }

  return (
    <aside className="arcade-panel flex h-full flex-col rounded-xl p-3">
      {/* Console Header */}
      <div className="mb-3 flex items-center justify-between border-b border-slate-800 pb-2">
        <div>
          <div className="flex items-center gap-1.5">
            <Zap className="h-4 w-4 text-fuchsia-400" />
            <h2 className="text-sm font-bold tracking-wide text-white">上帝整蛊控制台</h2>
          </div>
          <p className="font-mono text-[10px] uppercase tracking-widest text-slate-400">
            GOD-MODE CHAOS INJECTOR // KEYS [1-8]
          </p>
        </div>
        <span className="rounded-md border border-fuchsia-500/40 bg-fuchsia-500/15 px-2 py-0.5 font-mono text-[10px] font-bold text-fuchsia-300">
          LIVE ARMED
        </span>
      </div>

      <div className="flex flex-1 flex-col gap-3 overflow-y-auto pr-0.5">
        {/* 1. Physical Chaos */}
        <section className="rounded-xl border border-rose-500/25 bg-rose-950/10 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-rose-300">
              <Hand className="h-3.5 w-3.5" />
              <span>🖐 Physical Chaos</span>
            </div>
            <span className="font-mono text-[10px] text-rose-300/70">MUJOCO FORCE</span>
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            <ChaosSkillButton
              skill="object_nudge"
              icon={<Move className="h-3.5 w-3.5 text-rose-300" />}
              isActive={activeChaos.includes('object_nudge')}
              cooldownPct={getCooldownPct('object_nudge')}
              onClick={() => onTriggerChaos('object_nudge')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-rose-400/60 hover:bg-rose-500/10"
            />
            <ChaosSkillButton
              skill="arm_kick"
              icon={<Sparkles className="h-3.5 w-3.5 text-orange-300" />}
              isActive={activeChaos.includes('arm_kick')}
              cooldownPct={getCooldownPct('arm_kick')}
              onClick={() => onTriggerChaos('arm_kick')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-orange-400/60 hover:bg-orange-500/10"
            />
            <ChaosSkillButton
              skill="friction_glitch"
              icon={<Droplets className="h-3.5 w-3.5 text-cyan-300" />}
              isActive={activeChaos.includes('friction_glitch')}
              cooldownPct={getCooldownPct('friction_glitch')}
              onClick={() => onTriggerChaos('friction_glitch')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-cyan-400/60 hover:bg-cyan-500/10"
            />
          </div>
        </section>

        {/* 2. Visual Chaos */}
        <section className="rounded-xl border border-violet-500/25 bg-violet-950/10 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-violet-300">
              <Camera className="h-3.5 w-3.5" />
              <span>📷 Visual Chaos</span>
            </div>
            <span className="font-mono text-[10px] text-violet-300/70">RGB CORRUPTION</span>
          </div>
          <div className="grid grid-cols-1 gap-1.5">
            <ChaosSkillButton
              skill="lighting_shift"
              icon={<SunDim className="h-3.5 w-3.5 text-violet-300" />}
              isActive={activeChaos.includes('lighting_shift')}
              cooldownPct={getCooldownPct('lighting_shift')}
              onClick={() => onTriggerChaos('lighting_shift')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-violet-400/60 hover:bg-violet-500/10"
            />
            <ChaosSkillButton
              skill="camera_shake"
              icon={<Vibrate className="h-3.5 w-3.5 text-sky-300" />}
              isActive={activeChaos.includes('camera_shake')}
              cooldownPct={getCooldownPct('camera_shake')}
              onClick={() => onTriggerChaos('camera_shake')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-sky-400/60 hover:bg-sky-500/10"
            />
            <ChaosSkillButton
              skill="occlusion_patch"
              icon={<EyeOff className="h-3.5 w-3.5 text-amber-300" />}
              isActive={activeChaos.includes('occlusion_patch')}
              cooldownPct={getCooldownPct('occlusion_patch')}
              onClick={() => onTriggerChaos('occlusion_patch')}
              accentClass="border-slate-700/80 bg-slate-900/80 hover:border-amber-400/60 hover:bg-amber-500/10"
            />
          </div>
        </section>

        {/* 3. Systems Chaos */}
        <section className="rounded-xl border border-amber-500/25 bg-amber-950/10 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-amber-300">
              <Clock className="h-3.5 w-3.5" />
              <span>⏱ Systems Chaos</span>
            </div>
            <span className="font-mono text-[10px] text-amber-300/70">LOOP & SENSOR</span>
          </div>

          <div className="space-y-2.5">
            {/* Latency Slider (0-300ms) + Spike Trigger */}
            <div className="rounded-lg border border-slate-700/80 bg-slate-900/80 p-2.5">
              <div className="mb-1.5 flex items-center justify-between font-mono text-[11px]">
                <span className="font-semibold text-slate-200">延迟注入 (latency_inject)</span>
                <span
                  className={`font-bold ${
                    latencyMs >= 160
                      ? 'text-rose-400'
                      : latencyMs > 0
                        ? 'text-amber-300'
                        : 'text-slate-400'
                  }`}
                >
                  +{latencyMs} ms
                </span>
              </div>
              <input
                type="range"
                min={0}
                max={300}
                step={10}
                value={latencyMs}
                onChange={(e) => onChangeLatency(Number(e.target.value), false)}
                onMouseUp={(e) => onChangeLatency(Number(e.currentTarget.value), true)}
                onTouchEnd={(e) => onChangeLatency(Number(e.currentTarget.value), true)}
                className="cyber-slider cyber-slider-amber w-full"
              />
              <div className="mt-1.5 flex items-center justify-between font-mono text-[10px] text-slate-500">
                <span>0ms</span>
                <span>150ms</span>
                <button
                  type="button"
                  onClick={() => onTriggerChaos('latency_inject')}
                  className="rounded border border-amber-400/50 bg-amber-500/15 px-1.5 py-0.5 font-semibold text-amber-200 transition hover:bg-amber-500/25 active:scale-95"
                  title="Inject a timed latency spike burst (Shortcut: 7)"
                >
                  ⚡ SPIKE [7]
                </button>
              </div>
            </div>

            <ChaosSkillButton
              skill="wrist_blackout"
              icon={<VideoOff className="h-3.5 w-3.5 text-red-400" />}
              isActive={activeChaos.includes('wrist_blackout')}
              cooldownPct={getCooldownPct('wrist_blackout')}
              onClick={() => onTriggerChaos('wrist_blackout')}
              accentClass="border-red-500/35 bg-red-950/20 hover:border-red-400/75 hover:bg-red-500/15"
            />
          </div>
        </section>

        {/* 4. Brain Downgrade */}
        <section className="rounded-xl border border-fuchsia-500/25 bg-fuchsia-950/10 p-2.5">
          <div className="mb-2 flex items-center justify-between">
            <div className="flex items-center gap-1.5 font-mono text-[11px] font-bold uppercase tracking-wider text-fuchsia-300">
              <Brain className="h-3.5 w-3.5" />
              <span>🧠 Brain Downgrade</span>
            </div>
            <span className="font-mono text-[10px] text-fuchsia-300/70">QUANT & FLOW</span>
          </div>

          {/* Precision Hot-Swap Group */}
          <div className="mb-2.5">
            <div className="mb-1.5 flex items-center justify-between font-mono text-[10px] text-slate-400">
              <span>PRECISION HOT-SWAP</span>
              <span className="text-fuchsia-300">{precision.toUpperCase()}</span>
            </div>
            <div className="grid grid-cols-4 gap-1.5">
              {PRECISION_OPTIONS.map((opt) => {
                const selected = precision === opt.id
                return (
                  <button
                    key={opt.id}
                    type="button"
                    onClick={() => onChangePrecision(opt.id)}
                    className={`flex flex-col items-center rounded-lg border px-1.5 py-1.5 font-mono transition active:scale-95 ${
                      selected
                        ? `${opt.colorClass} shadow-[0_0_14px_rgba(232,121,249,0.22)]`
                        : 'border-slate-700/80 bg-slate-900/80 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                    }`}
                  >
                    <span className="text-xs font-bold">{opt.label}</span>
                    <span className="text-[9px] opacity-75">{opt.tag}</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Denoise Steps Slider (1-10) */}
          <div className="rounded-lg border border-slate-700/80 bg-slate-900/80 p-2.5">
            <div className="mb-1.5 flex items-center justify-between font-mono text-[11px]">
              <span className="font-semibold text-slate-200">去噪步数 (Denoise)</span>
              <span
                className={`font-bold ${
                  denoiseSteps <= 2
                    ? 'text-fuchsia-300'
                    : denoiseSteps <= 5
                      ? 'text-cyan-300'
                      : 'text-emerald-300'
                }`}
              >
                {denoiseSteps} {denoiseSteps === 1 ? 'step' : 'steps'}
              </span>
            </div>
            <input
              type="range"
              min={1}
              max={10}
              step={1}
              value={denoiseSteps}
              onChange={(e) => onChangeDenoiseSteps(Number(e.target.value))}
              className="cyber-slider cyber-slider-fuchsia w-full"
            />
            <div className="mt-1 flex justify-between font-mono text-[9px] text-slate-500">
              <span>1 (FAST / NOISY)</span>
              <span>5</span>
              <span>10 (SMOOTH)</span>
            </div>
          </div>
        </section>
      </div>
    </aside>
  )
}
