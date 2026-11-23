import { useCallback, useEffect, useRef, useState } from 'react'
import { DemoSimulationEngine } from '../sim/demoEngine'
import {
  CHAOS_SKILL_META,
  type ChaosMessage,
  type ChaosParamsMap,
  type ChaosSkill,
  type ClientMessage,
  type ConfigMessage,
  type ConnectionMode,
  type ConnectionState,
  type ControlAction,
  type EventCategory,
  type EventLogEntry,
  type FrameMessage,
  type FrameStatus,
  type Precision,
  PRESET_PROMPTS,
  type ServerMessage,
  deriveHealthUrl,
  formatTimestamp,
} from '../types/protocol'
import {
  downloadBase64Gif,
  encodeFramesToGifBlob,
  triggerBlobDownload,
} from '../utils/gifEncoder'

const DEFAULT_WS_URL = 'ws://localhost:8000/ws/session'
const WS_STORAGE_KEY = 'break_my_vla_ws_url'
const MAX_LOGS = 80
const MAX_LATENCY_HISTORY = 32

function loadSavedWsUrl(): string {
  try {
    return localStorage.getItem(WS_STORAGE_KEY) || DEFAULT_WS_URL
  } catch {
    return DEFAULT_WS_URL
  }
}

function randomSignedOffset(minAbs: number, maxAbs: number): number {
  const sign = Math.random() < 0.5 ? -1 : 1
  const val = minAbs + Math.random() * (maxAbs - minAbs)
  return Number((sign * val).toFixed(3))
}

export function buildDefaultChaosParams<K extends ChaosSkill>(
  skill: K,
  currentLatencyMs = 160,
): ChaosParamsMap[K] {
  switch (skill) {
    case 'object_nudge':
      return {
        dx: randomSignedOffset(0.14, 0.28),
        dy: randomSignedOffset(0.1, 0.22),
        dz: 0.0,
      } as ChaosParamsMap[K]
    case 'arm_kick':
      return {
        dx: randomSignedOffset(0.18, 0.32),
        dy: randomSignedOffset(0.1, 0.22),
        dz: Number((0.08 + Math.random() * 0.1).toFixed(3)),
      } as ChaosParamsMap[K]
    case 'friction_glitch':
      return {
        duration_steps: 45,
        multiplier: 0.1,
      } as ChaosParamsMap[K]
    case 'lighting_shift':
      return {
        duration_steps: 40,
        brightness: Number((0.32 + Math.random() * 0.35).toFixed(2)),
        saturation: Number((0.55 + Math.random() * 0.9).toFixed(2)),
      } as ChaosParamsMap[K]
    case 'camera_shake':
      return {
        duration_steps: 36,
        amplitude_px: 16,
      } as ChaosParamsMap[K]
    case 'occlusion_patch':
      return {
        duration_steps: 45,
        x: Number((0.18 + Math.random() * 0.34).toFixed(2)),
        y: Number((0.16 + Math.random() * 0.32).toFixed(2)),
        width: 0.3,
        height: 0.26,
      } as ChaosParamsMap[K]
    case 'latency_inject':
      return {
        duration_steps: 40,
        latency_ms: Math.max(80, Math.min(300, Math.round(currentLatencyMs || 180))),
      } as ChaosParamsMap[K]
    case 'wrist_blackout':
      return {
        duration_steps: 40,
      } as ChaosParamsMap[K]
  }
}

export interface LatencySample {
  step: number
  inferMs: number
}

export interface UseVlaSessionResult {
  wsUrl: string
  mode: ConnectionMode
  connectionState: ConnectionState
  modelName: string
  frame: FrameMessage | null
  latencyHistory: LatencySample[]
  logs: EventLogEntry[]
  prompt: string
  latencyMs: number
  precision: Precision
  denoiseSteps: number
  isExportingGif: boolean
  cooldowns: Partial<Record<ChaosSkill, { startedAt: number; endsAt: number }>>
  connectLive: (targetUrl?: string) => void
  switchToDemo: () => void
  triggerChaos: <K extends ChaosSkill>(skill: K, customParams?: ChaosParamsMap[K]) => void
  updatePrecision: (precision: Precision) => void
  updateDenoiseSteps: (steps: number) => void
  updateLatencyMs: (ms: number, emitLog?: boolean) => void
  updatePrompt: (newPrompt: string) => void
  sendControl: (action: ControlAction) => void
  clearLogs: () => void
}

export function useVlaSession(): UseVlaSessionResult {
  const [wsUrl, setWsUrl] = useState<string>(loadSavedWsUrl)
  const [mode, setMode] = useState<ConnectionMode>('demo')
  const [connectionState, setConnectionState] = useState<ConnectionState>('connecting')
  const [modelName, setModelName] = useState<string>('demo-vla-sim (local)')
  const [frame, setFrame] = useState<FrameMessage | null>(null)
  const [latencyHistory, setLatencyHistory] = useState<LatencySample[]>([])
  const [logs, setLogs] = useState<EventLogEntry[]>([])
  const [prompt, setPrompt] = useState<string>(PRESET_PROMPTS[0])
  const [latencyMs, setLatencyMs] = useState<number>(0)
  const [precision, setPrecision] = useState<Precision>('bf16')
  const [denoiseSteps, setDenoiseSteps] = useState<number>(8)
  const [isExportingGif, setIsExportingGif] = useState<boolean>(false)
  const [cooldowns, setCooldowns] = useState<
    Partial<Record<ChaosSkill, { startedAt: number; endsAt: number }>>
  >({})

  const wsRef = useRef<WebSocket | null>(null)
  const demoEngineRef = useRef<DemoSimulationEngine | null>(null)
  const demoTimerRef = useRef<number | null>(null)
  const connectTimeoutRef = useRef<number | null>(null)
  const lastStatusRef = useRef<FrameStatus | null>(null)
  const modeRef = useRef<ConnectionMode>('demo')
  const logCounterRef = useRef<number>(0)

  const appendLog = useCallback(
    (
      category: EventCategory,
      title: string,
      detail: string,
      severity: EventLogEntry['severity'] = 'info',
    ) => {
      logCounterRef.current += 1
      const entry: EventLogEntry = {
        id: `${Date.now()}-${logCounterRef.current}`,
        timestamp: formatTimestamp(),
        category,
        title,
        detail,
        severity,
      }
      setLogs((prev) => [entry, ...prev].slice(0, MAX_LOGS))
    },
    [],
  )

  const ingestFrame = useCallback(
    (incoming: FrameMessage) => {
      setFrame(incoming)
      setPrecision(incoming.precision)
      setDenoiseSteps(incoming.denoise_steps)
      if (incoming.model) {
        setModelName(incoming.model)
      }
      setLatencyHistory((prev) => {
        const next = [...prev, { step: incoming.step, inferMs: incoming.infer_ms }]
        return next.length > MAX_LATENCY_HISTORY
          ? next.slice(next.length - MAX_LATENCY_HISTORY)
          : next
      })

      if (lastStatusRef.current !== incoming.status) {
        const prev = lastStatusRef.current
        lastStatusRef.current = incoming.status
        if (prev !== null) {
          const severity =
            incoming.status === 'success'
              ? 'success'
              : incoming.status === 'policy_error'
                ? 'danger'
                : 'info'
          appendLog(
            'status',
            `STATUS -> ${incoming.status.toUpperCase()}`,
            `step=${incoming.step} progress=${Math.round(incoming.task_progress * 100)}%`,
            severity,
          )
        }
      }
    },
    [appendLog],
  )

  const stopDemoLoop = useCallback(() => {
    if (demoTimerRef.current !== null) {
      window.clearInterval(demoTimerRef.current)
      demoTimerRef.current = null
    }
  }, [])

  const startDemoLoop = useCallback(
    (reason: 'fallback' | 'manual') => {
      stopDemoLoop()
      if (!demoEngineRef.current) {
        demoEngineRef.current = new DemoSimulationEngine()
      }
      modeRef.current = 'demo'
      setMode('demo')
      setConnectionState(reason === 'fallback' ? 'demo_fallback' : 'demo_manual')
      setModelName('demo-vla-sim (local)')

      const initialFrame = demoEngineRef.current.tick()
      ingestFrame(initialFrame)

      demoTimerRef.current = window.setInterval(() => {
        if (!demoEngineRef.current) return
        const nextFrame = demoEngineRef.current.tick()
        ingestFrame(nextFrame)
      }, 60)
    },
    [ingestFrame, stopDemoLoop],
  )

  const closeActiveSocket = useCallback(() => {
    if (connectTimeoutRef.current !== null) {
      window.clearTimeout(connectTimeoutRef.current)
      connectTimeoutRef.current = null
    }
    if (wsRef.current) {
      wsRef.current.onopen = null
      wsRef.current.onmessage = null
      wsRef.current.onerror = null
      wsRef.current.onclose = null
      try {
        wsRef.current.close()
      } catch {
        // ignore close errors
      }
      wsRef.current = null
    }
  }, [])

  const probeBackendHealth = useCallback(async (url: string) => {
    const healthUrl = deriveHealthUrl(url)
    if (!healthUrl) return
    try {
      const controller = new AbortController()
      const timer = window.setTimeout(() => controller.abort(), 1800)
      const res = await fetch(healthUrl, { signal: controller.signal })
      window.clearTimeout(timer)
      if (!res.ok) return
      const data = (await res.json()) as { model?: string; policy_model?: string }
      const detected = data.model || data.policy_model
      if (detected) {
        setModelName(detected)
      }
    } catch {
      // Optional health probe; silently ignore CORS/missing route
    }
  }, [])

  const connectLive = useCallback(
    (targetUrl?: string) => {
      const resolvedUrl = (targetUrl ?? wsUrl).trim() || DEFAULT_WS_URL
      setWsUrl(resolvedUrl)
      try {
        localStorage.setItem(WS_STORAGE_KEY, resolvedUrl)
      } catch {
        // ignore storage errors
      }

      closeActiveSocket()
      setConnectionState('connecting')
      appendLog('system', 'CONNECTING WS', resolvedUrl, 'info')

      let opened = false
      let socket: WebSocket
      try {
        socket = new WebSocket(resolvedUrl)
      } catch {
        appendLog(
          'system',
          'INVALID WS URL -> LOCAL DEMO',
          `Cannot open ${resolvedUrl}; running local simulator`,
          'warn',
        )
        startDemoLoop('fallback')
        return
      }

      wsRef.current = socket

      connectTimeoutRef.current = window.setTimeout(() => {
        if (!opened) {
          closeActiveSocket()
          appendLog(
            'system',
            'BACKEND TIMEOUT -> LOCAL DEMO',
            `No response from ${resolvedUrl}; auto-fallback active`,
            'warn',
          )
          startDemoLoop('fallback')
        }
      }, 2200)

      socket.onopen = () => {
        opened = true
        if (connectTimeoutRef.current !== null) {
          window.clearTimeout(connectTimeoutRef.current)
          connectTimeoutRef.current = null
        }
        stopDemoLoop()
        modeRef.current = 'live'
        setMode('live')
        setConnectionState('connected')
        setModelName('pi05_libero (live)')
        lastStatusRef.current = null
        appendLog('system', 'LIVE WS CONNECTED', resolvedUrl, 'success')
        void probeBackendHealth(resolvedUrl)
      }

      socket.onmessage = (event: MessageEvent) => {
        try {
          const msg = JSON.parse(String(event.data)) as ServerMessage
          if (msg.type === 'frame') {
            ingestFrame(msg)
          } else if (msg.type === 'gif') {
            setIsExportingGif(false)
            downloadBase64Gif(msg.gif_b64)
            appendLog('control', 'GIF EXPORTED', 'Downloaded live episode replay GIF', 'success')
          } else if (msg.type === 'error') {
            setIsExportingGif(false)
            appendLog('error', `SERVER ERROR [${msg.code}]`, msg.message, 'danger')
          }
        } catch {
          appendLog('error', 'MALFORMED WS FRAME', 'Failed to parse JSON message', 'danger')
        }
      }

      socket.onerror = () => {
        // Handled cleanly by onclose
      }

      socket.onclose = () => {
        if (connectTimeoutRef.current !== null) {
          window.clearTimeout(connectTimeoutRef.current)
          connectTimeoutRef.current = null
        }
        wsRef.current = null
        if (!opened) {
          appendLog(
            'system',
            'BACKEND OFFLINE -> LOCAL DEMO',
            `${resolvedUrl} unavailable; local simulator enabled`,
            'warn',
          )
        } else {
          appendLog(
            'system',
            'WS DISCONNECTED -> LOCAL DEMO',
            'Live stream closed; switched to Local Demo fallback',
            'warn',
          )
        }
        startDemoLoop('fallback')
      }
    },
    [
      appendLog,
      closeActiveSocket,
      ingestFrame,
      probeBackendHealth,
      startDemoLoop,
      stopDemoLoop,
      wsUrl,
    ],
  )

  const switchToDemo = useCallback(() => {
    closeActiveSocket()
    appendLog('system', 'LOCAL DEMO MODE', 'Switched manually to local simulator', 'info')
    startDemoLoop('manual')
  }, [appendLog, closeActiveSocket, startDemoLoop])

  useEffect(() => {
    connectLive(loadSavedWsUrl())
    return () => {
      closeActiveSocket()
      stopDemoLoop()
    }
    // Mount-only initial connection attempt
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const sendWireMessage = useCallback((payload: ClientMessage): boolean => {
    const ws = wsRef.current
    if (modeRef.current === 'live' && ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload))
      return true
    }
    return false
  }, [])

  const triggerChaos = useCallback(
    <K extends ChaosSkill>(skill: K, customParams?: ChaosParamsMap[K]) => {
      const meta = CHAOS_SKILL_META[skill]
      const now = Date.now()
      const params = customParams ?? buildDefaultChaosParams(skill, latencyMs)
      const message: ChaosMessage<K> = {
        type: 'chaos',
        skill,
        params,
      }

      setCooldowns((prev) => ({
        ...prev,
        [skill]: { startedAt: now, endsAt: now + meta.cooldownMs },
      }))

      if (modeRef.current === 'live') {
        sendWireMessage(message)
      } else {
        demoEngineRef.current?.injectChaos(skill, params)
      }

      const paramSummary = Object.entries(params as unknown as Record<string, number>)
        .map(([k, v]) => `${k}=${v}`)
        .join(' ')
      appendLog('chaos', `CHAOS // ${meta.labelZh} (${skill})`, paramSummary, 'warn')
    },
    [appendLog, latencyMs, sendWireMessage],
  )

  const sendConfig = useCallback(
    (cfg: Omit<ConfigMessage, 'type'>, logTitle?: string, logDetail?: string) => {
      const payload: ConfigMessage = { type: 'config', ...cfg }
      if (modeRef.current === 'live') {
        sendWireMessage(payload)
      } else {
        demoEngineRef.current?.configure(cfg)
      }
      if (logTitle) {
        appendLog('config', logTitle, logDetail ?? '', 'info')
      }
    },
    [appendLog, sendWireMessage],
  )

  const updatePrecision = useCallback(
    (nextPrecision: Precision) => {
      setPrecision(nextPrecision)
      sendConfig(
        { precision: nextPrecision },
        `PRECISION -> ${nextPrecision.toUpperCase()}`,
        'Runtime quantization hot-swap dispatched',
      )
    },
    [sendConfig],
  )

  const updateDenoiseSteps = useCallback(
    (steps: number) => {
      const clamped = Math.max(1, Math.min(10, Math.round(steps)))
      setDenoiseSteps(clamped)
      sendConfig(
        { denoise_steps: clamped },
        `DENOISE STEPS -> ${clamped}`,
        `Flow-matching trajectory steps=${clamped}`,
      )
    },
    [sendConfig],
  )

  const updateLatencyMs = useCallback(
    (ms: number, emitLog = false) => {
      const clamped = Math.max(0, Math.min(300, Math.round(ms)))
      setLatencyMs(clamped)
      sendConfig(
        { latency_ms: clamped },
        emitLog ? `LATENCY INJECT -> ${clamped}ms` : undefined,
        emitLog ? 'Systems loop delay updated' : undefined,
      )
    },
    [sendConfig],
  )

  const updatePrompt = useCallback(
    (newPrompt: string) => {
      const trimmed = newPrompt.trim()
      if (!trimmed) return
      setPrompt(trimmed)
      sendConfig({ prompt: trimmed }, 'PROMPT UPDATED', trimmed)
    },
    [sendConfig],
  )

  const sendControl = useCallback(
    (action: ControlAction) => {
      if (action === 'export_gif') {
        setIsExportingGif(true)
        if (modeRef.current === 'live') {
          sendWireMessage({ type: 'control', action: 'export_gif' })
          appendLog('control', 'EXPORT GIF REQUESTED', 'Waiting for backend replay buffer...', 'info')
          window.setTimeout(() => setIsExportingGif(false), 6000)
        } else {
          try {
            const frames = demoEngineRef.current?.getRecordedGifFrames() ?? []
            if (frames.length < 2) {
              appendLog('error', 'GIF UNAVAILABLE', 'Collect a few more frames before exporting', 'warn')
            } else {
              const blob = encodeFramesToGifBlob(frames, 8)
              const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
              triggerBlobDownload(blob, `break-my-vla-demo-${stamp}.gif`)
              appendLog(
                'control',
                'DEMO GIF EXPORTED',
                `Encoded ${frames.length} frames (${Math.round(blob.size / 1024)} KB)`,
                'success',
              )
            }
          } catch {
            appendLog('error', 'GIF EXPORT FAILED', 'Local encoder error', 'danger')
          } finally {
            setIsExportingGif(false)
          }
        }
        return
      }

      if (modeRef.current === 'live') {
        sendWireMessage({ type: 'control', action })
      } else if (demoEngineRef.current) {
        if (action === 'reset') {
          demoEngineRef.current.reset()
          setLatencyHistory([])
          lastStatusRef.current = null
        } else if (action === 'pause') {
          demoEngineRef.current.setPaused(true)
        } else if (action === 'resume') {
          demoEngineRef.current.setPaused(false)
        }
      }

      appendLog('control', `CONTROL // ${action.toUpperCase()}`, `Dispatched (${modeRef.current})`, 'info')
    },
    [appendLog, sendWireMessage],
  )

  const clearLogs = useCallback(() => {
    setLogs([])
  }, [])

  return {
    wsUrl,
    mode,
    connectionState,
    modelName,
    frame,
    latencyHistory,
    logs,
    prompt,
    latencyMs,
    precision,
    denoiseSteps,
    isExportingGif,
    cooldowns,
    connectLive,
    switchToDemo,
    triggerChaos,
    updatePrecision,
    updateDenoiseSteps,
    updateLatencyMs,
    updatePrompt,
    sendControl,
    clearLogs,
  }
}
