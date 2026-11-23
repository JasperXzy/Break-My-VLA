import { useEffect, useRef } from 'react'
import { AlertTriangle, Camera, CheckCircle2, PauseCircle, VideoOff } from 'lucide-react'
import {
  CHAOS_SKILL_META,
  type ChaosSkill,
  type FrameMessage,
  toImageDataUri,
} from '../types/protocol'

interface CameraViewportProps {
  frame: FrameMessage | null
}

/** High-FPS animated TV static snow overlay rendered during wrist_blackout. */
function StaticSnowCanvas() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    let rafId = 0
    const width = 200
    const height = 130
    canvas.width = width
    canvas.height = height
    const imgData = ctx.createImageData(width, height)
    const buf = imgData.data

    const renderNoise = () => {
      for (let i = 0; i < buf.length; i += 4) {
        const shade = Math.random() > 0.12 ? Math.floor(Math.random() * 235) : 255
        buf[i] = shade
        buf[i + 1] = shade
        buf[i + 2] = Math.min(255, shade + 12)
        buf[i + 3] = 245
      }
      ctx.putImageData(imgData, 0, 0)

      // Horizontal VHS / signal tear bars
      const tearY = Math.floor(Math.random() * height)
      ctx.fillStyle = 'rgba(239, 68, 68, 0.28)'
      ctx.fillRect(0, tearY, width, 4 + Math.random() * 8)
      ctx.fillStyle = 'rgba(255, 255, 255, 0.35)'
      ctx.fillRect(0, (tearY + 26) % height, width, 2)

      rafId = window.requestAnimationFrame(renderNoise)
    }

    rafId = window.requestAnimationFrame(renderNoise)
    return () => window.cancelAnimationFrame(rafId)
  }, [])

  return (
    <div className="absolute inset-0 z-20 flex flex-col items-center justify-center overflow-hidden bg-slate-950">
      <canvas
        ref={canvasRef}
        className="h-full w-full object-cover opacity-85 mix-blend-screen"
      />
      <div className="crt-scanlines pointer-events-none absolute inset-0" />
      <div className="animate-glitch pointer-events-none absolute inset-0 flex flex-col items-center justify-center bg-red-950/35 px-4 text-center">
        <div className="mb-1.5 flex items-center gap-2 rounded-md border border-red-400/70 bg-red-950/90 px-3 py-1 shadow-[0_0_24px_rgba(239,68,68,0.65)]">
          <VideoOff className="h-4 w-4 text-red-400" />
          <span className="font-mono text-xs font-bold tracking-widest text-red-200">
            SIGNAL LOST // NO SYNC
          </span>
        </div>
        <span className="font-mono text-[10px] tracking-wider text-red-300/90">
          WRIST_CAM DROPOUT INJECTED
        </span>
      </div>
    </div>
  )
}

function HudCornerBrackets({ accent = 'cyan' }: { accent?: 'cyan' | 'red' }) {
  const borderClass = accent === 'red' ? 'border-red-400/80' : 'border-cyan-400/50'
  return (
    <div className="pointer-events-none absolute inset-2 z-10">
      <span className={`absolute left-0 top-0 h-3 w-3 border-l-2 border-t-2 ${borderClass}`} />
      <span className={`absolute right-0 top-0 h-3 w-3 border-r-2 border-t-2 ${borderClass}`} />
      <span className={`absolute bottom-0 left-0 h-3 w-3 border-b-2 border-l-2 ${borderClass}`} />
      <span className={`absolute bottom-0 right-0 h-3 w-3 border-b-2 border-r-2 ${borderClass}`} />
    </div>
  )
}

export function CameraViewport({ frame }: CameraViewportProps) {
  const activeChaos: ChaosSkill[] = frame?.active_chaos ?? []
  const wristBlackout = activeChaos.includes('wrist_blackout')
  const mainSrc = frame ? toImageDataUri(frame.rgb_main_b64) : ''
  const wristSrc = frame ? toImageDataUri(frame.rgb_wrist_b64) : ''
  const status = frame?.status ?? 'running'

  const mainVisualChaos = activeChaos.filter((s) => s !== 'wrist_blackout')

  return (
    <div className="grid grid-cols-1 gap-3 xl:grid-cols-12">
      {/* Main View (Third-Person Camera) */}
      <div className="arcade-panel relative overflow-hidden rounded-xl xl:col-span-7">
        <div className="flex items-center justify-between border-b border-slate-800/90 bg-slate-950/70 px-3 py-1.5">
          <div className="flex items-center gap-2 font-mono text-[11px] font-semibold tracking-wider text-cyan-300">
            <Camera className="h-3.5 w-3.5" />
            <span>MAIN VIEW // CAM_0 (3RD-PERSON)</span>
          </div>
          <div className="flex items-center gap-2 font-mono text-[10px]">
            <span className="flex items-center gap-1 text-emerald-300">
              <span className="h-1.5 w-1.5 animate-pulse rounded-full bg-emerald-400" />
              REC
            </span>
            <span className="text-slate-400">
              STEP {String(frame?.step ?? 0).padStart(4, '0')}
            </span>
          </div>
        </div>

        <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-950">
          {mainSrc ? (
            <img
              src={mainSrc}
              alt="Main third-person robot view"
              className="h-full w-full object-cover"
              draggable={false}
            />
          ) : (
            <div className="flex h-full items-center justify-center font-mono text-xs text-slate-500">
              AWAITING VIDEO STREAM...
            </div>
          )}

          <div className="crt-scanlines pointer-events-none absolute inset-0 opacity-40" />
          <HudCornerBrackets accent={mainVisualChaos.length > 0 ? 'red' : 'cyan'} />

          {/* Floating active chaos warning badges on main frame */}
          {mainVisualChaos.length > 0 && (
            <div className="pointer-events-none absolute left-3 top-3 z-20 flex flex-wrap gap-1.5">
              {mainVisualChaos.map((skill) => (
                <span
                  key={skill}
                  className="flex items-center gap-1 rounded border border-rose-400/70 bg-rose-950/85 px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider text-rose-200 shadow-[0_0_12px_rgba(244,63,94,0.45)]"
                >
                  <AlertTriangle className="h-3 w-3 text-rose-400" />
                  {CHAOS_SKILL_META[skill].labelEn}
                </span>
              ))}
            </div>
          )}

          {/* Status overlays */}
          {status === 'paused' && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-slate-950/55 backdrop-blur-[1px]">
              <div className="flex items-center gap-2 rounded-lg border border-amber-400/60 bg-slate-950/90 px-4 py-2 font-mono text-xs font-bold tracking-widest text-amber-300 shadow-lg">
                <PauseCircle className="h-4 w-4" />
                SIMULATION PAUSED
              </div>
            </div>
          )}

          {status === 'success' && (
            <div className="pointer-events-none absolute inset-0 z-20 flex items-center justify-center bg-emerald-950/30">
              <div className="flex items-center gap-2 rounded-lg border border-emerald-400/70 bg-slate-950/90 px-4 py-2 font-mono text-xs font-bold tracking-widest text-emerald-300 shadow-[0_0_24px_rgba(16,185,129,0.45)]">
                <CheckCircle2 className="h-4 w-4" />
                TASK COMPLETED // PRESS RESET OR NUDGE!
              </div>
            </div>
          )}

          {status === 'policy_error' && (
            <div className="absolute inset-0 z-20 flex items-center justify-center bg-rose-950/55">
              <div className="flex items-center gap-2 rounded-lg border border-rose-400/70 bg-slate-950/95 px-4 py-2 font-mono text-xs font-bold tracking-widest text-rose-300">
                <AlertTriangle className="h-4 w-4" />
                POLICY SERVER ERROR
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Wrist Cam (End-Effector Camera) */}
      <div
        className={`arcade-panel relative overflow-hidden rounded-xl transition xl:col-span-5 ${
          wristBlackout ? 'border-red-500/70 shadow-[0_0_20px_rgba(239,68,68,0.28)]' : ''
        }`}
      >
        <div className="flex items-center justify-between border-b border-slate-800/90 bg-slate-950/70 px-3 py-1.5">
          <div className="flex items-center gap-2 font-mono text-[11px] font-semibold tracking-wider text-fuchsia-300">
            <Camera className="h-3.5 w-3.5" />
            <span>WRIST CAM // CAM_1 (EEF)</span>
          </div>
          <span
            className={`rounded px-1.5 py-0.5 font-mono text-[10px] font-bold ${
              wristBlackout
                ? 'bg-red-500/20 text-red-300'
                : 'bg-emerald-500/15 text-emerald-300'
            }`}
          >
            {wristBlackout ? 'DROPOUT' : 'LOCKED'}
          </span>
        </div>

        <div className="relative aspect-[16/10] w-full overflow-hidden bg-slate-950">
          {wristSrc ? (
            <img
              src={wristSrc}
              alt="Wrist camera robot view"
              className="h-full w-full object-cover"
              draggable={false}
            />
          ) : (
            <div className="flex h-full items-center justify-center font-mono text-xs text-slate-500">
              AWAITING WRIST STREAM...
            </div>
          )}

          {wristBlackout && <StaticSnowCanvas />}

          <div className="crt-scanlines pointer-events-none absolute inset-0 opacity-40" />
          <HudCornerBrackets accent={wristBlackout ? 'red' : 'cyan'} />

          {!wristBlackout && (
            <div className="pointer-events-none absolute bottom-2.5 right-3 z-10 rounded bg-slate-950/75 px-1.5 py-0.5 font-mono text-[10px] text-slate-300">
              PREC: {(frame?.precision ?? 'bf16').toUpperCase()} · {frame?.denoise_steps ?? 8} STEPS
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
