import { useEffect, useState } from 'react'
import { BenchmarkGallery } from './components/BenchmarkGallery'
import { CameraViewport } from './components/CameraViewport'
import { ChaosConsole } from './components/ChaosConsole'
import { EventLogBar } from './components/EventLogBar'
import { TelemetryDashboard } from './components/TelemetryDashboard'
import { TopStatusBar } from './components/TopStatusBar'
import { useVlaSession } from './hooks/useVlaSession'
import type { AppView, ChaosSkill } from './types/protocol'
import { parseArenaHash } from './utils/arenaHash'

const HOTKEY_TO_SKILL: Record<string, ChaosSkill> = {
  '1': 'object_nudge',
  '2': 'arm_kick',
  '3': 'friction_glitch',
  '4': 'lighting_shift',
  '5': 'camera_shake',
  '6': 'occlusion_patch',
  '7': 'latency_inject',
  '8': 'wrist_blackout',
}

function getInitialView(): AppView {
  if (typeof window === 'undefined') return 'arcade'
  const parsed = parseArenaHash(window.location.hash)
  const params = new URLSearchParams(window.location.search)
  if (parsed.view === 'gallery' || params.get('view') === 'gallery') {
    return 'gallery'
  }
  return 'arcade'
}

export function App() {
  const [activeView, setActiveView] = useState<AppView>(getInitialView)
  const session = useVlaSession()
  const { frame, triggerChaos, sendControl } = session
  const currentStatus = frame?.status ?? 'running'

  const handleChangeView = (view: AppView) => {
    setActiveView(view)
    if (typeof window !== 'undefined') {
      if (view === 'gallery') {
        if (!window.location.hash.toLowerCase().startsWith('#gallery')) {
          window.history.replaceState(null, '', '#gallery')
        }
      } else if (window.location.hash !== '#arcade') {
        window.history.replaceState(null, '', '#arcade')
      }
    }
  }

  useEffect(() => {
    const onHashChange = () => {
      const parsed = parseArenaHash(window.location.hash)
      if (parsed.hasExplicitView) {
        setActiveView(parsed.view)
      }
    }
    window.addEventListener('hashchange', onHashChange)
    return () => window.removeEventListener('hashchange', onHashChange)
  }, [])

  // Keyboard arcade hotkeys (active only on Live Arcade view)
  useEffect(() => {
    if (activeView !== 'arcade') return
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null
      if (
        target &&
        (target.tagName === 'INPUT' ||
          target.tagName === 'TEXTAREA' ||
          target.tagName === 'SELECT' ||
          target.isContentEditable)
      ) {
        return
      }
      if (e.metaKey || e.ctrlKey || e.altKey) return

      const skill = HOTKEY_TO_SKILL[e.key]
      if (skill) {
        e.preventDefault()
        triggerChaos(skill)
        return
      }

      if (e.code === 'Space') {
        e.preventDefault()
        sendControl(currentStatus === 'paused' ? 'resume' : 'pause')
      } else if (e.key.toLowerCase() === 'r') {
        e.preventDefault()
        sendControl('reset')
      }
    }

    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [activeView, currentStatus, sendControl, triggerChaos])

  return (
    <div className="cyber-grid-bg flex min-h-screen flex-col text-slate-100">
      <TopStatusBar
        activeView={activeView}
        onChangeView={handleChangeView}
        wsUrl={session.wsUrl}
        mode={session.mode}
        connectionState={session.connectionState}
        reconnectAttempt={session.reconnectAttempt}
        modelName={session.modelName}
        activeModelSpec={session.activeModelSpec}
        availableModels={session.availableModels}
        fps={frame?.fps ?? 0}
        inferMs={frame?.infer_ms ?? 0}
        step={frame?.step ?? 0}
        status={currentStatus}
        prompt={session.prompt}
        isExportingGif={session.isExportingGif}
        onConnectLive={session.connectLive}
        onSwitchToDemo={session.switchToDemo}
        onSelectModel={session.selectModel}
        onSubmitPrompt={session.updatePrompt}
        onReset={() => sendControl('reset')}
        onTogglePause={() =>
          sendControl(currentStatus === 'paused' ? 'resume' : 'pause')
        }
        onExportGif={() => sendControl('export_gif')}
      />

      {activeView === 'gallery' ? (
        <BenchmarkGallery />
      ) : (
        <>
          {/* Main Split Workspace */}
          <main className="mx-auto grid w-full max-w-[1720px] flex-1 grid-cols-1 gap-3 p-3 lg:grid-cols-12 lg:p-4">
            {/* Left Primary Viewport + Telemetry HUD */}
            <div className="flex flex-col gap-3 lg:col-span-8 xl:col-span-9">
              <CameraViewport frame={frame} />
              <TelemetryDashboard
                actionChunk={frame?.action_chunk ?? []}
                inferMs={frame?.infer_ms ?? 0}
                latencyHistory={session.latencyHistory}
                taskProgress={frame?.task_progress ?? 0}
                activeChaos={frame?.active_chaos ?? []}
                status={currentStatus}
              />
            </div>

            {/* Right God-Mode Chaos Console */}
            <div className="lg:col-span-4 xl:col-span-3">
              <ChaosConsole
                activeChaos={frame?.active_chaos ?? []}
                cooldowns={session.cooldowns}
                latencyMs={session.latencyMs}
                precision={session.precision}
                denoiseSteps={session.denoiseSteps}
                onTriggerChaos={(skill) => triggerChaos(skill)}
                onChangeLatency={session.updateLatencyMs}
                onChangePrecision={session.updatePrecision}
                onChangeDenoiseSteps={session.updateDenoiseSteps}
              />
            </div>
          </main>

          <EventLogBar logs={session.logs} onClear={session.clearLogs} />
        </>
      )}
    </div>
  )
}

export default App
