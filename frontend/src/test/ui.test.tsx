import { act, fireEvent, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import { App } from '../App'
import { BenchmarkGallery } from '../components/BenchmarkGallery'
import { CameraViewport } from '../components/CameraViewport'
import { ChaosConsole } from '../components/ChaosConsole'
import type { FrameMessage } from '../types/protocol'

function makeSampleFrame(overrides: Partial<FrameMessage> = {}): FrameMessage {
  return {
    type: 'frame',
    step: 12,
    fps: 19.4,
    infer_ms: 48.2,
    precision: 'bf16',
    denoise_steps: 10,
    rgb_main_b64: '/9j/4AAQSkZJRg==',
    rgb_wrist_b64: '/9j/4AAQSkZJRg==',
    action_chunk: [
      [0.45, -0.32, 0.12, 0.01, -0.02, 0.0, 0.9],
      [0.38, -0.25, 0.09, 0.0, 0.0, 0.0, 0.9],
    ],
    active_chaos: [],
    task_progress: 0.42,
    status: 'running',
    ...overrides,
  }
}

class MockLiveWebSocket {
  static CONNECTING = 0
  static OPEN = 1
  static CLOSING = 2
  static CLOSED = 3
  static instances: MockLiveWebSocket[] = []

  public readyState = MockLiveWebSocket.CONNECTING
  public sent: string[] = []
  public onopen: ((ev: Event) => void) | null = null
  public onmessage: ((ev: MessageEvent) => void) | null = null
  public onerror: ((ev: Event) => void) | null = null
  public onclose: ((ev: CloseEvent) => void) | null = null
  public url: string

  constructor(url: string) {
    this.url = url
    MockLiveWebSocket.instances.push(this)
  }

  send(data: string) {
    this.sent.push(data)
  }

  close() {
    this.readyState = MockLiveWebSocket.CLOSED
  }

  emitOpen() {
    this.readyState = MockLiveWebSocket.OPEN
    this.onopen?.(new Event('open'))
  }

  emitJson(payload: unknown) {
    this.onmessage?.(new MessageEvent('message', { data: JSON.stringify(payload) }))
  }

  emitUnexpectedClose() {
    this.readyState = MockLiveWebSocket.CLOSED
    this.onclose?.(new CloseEvent('close'))
  }
}

describe('CameraViewport & ChaosConsole components', () => {
  it('displays static snow SIGNAL LOST overlay during wrist_blackout', () => {
    render(
      <CameraViewport
        frame={makeSampleFrame({ active_chaos: ['wrist_blackout', 'camera_shake'] })}
      />,
    )
    expect(screen.getByText(/SIGNAL LOST \/\/ NO SYNC/i)).toBeInTheDocument()
    expect(screen.getByText('DROPOUT')).toBeInTheDocument()
    expect(screen.getByText(/CAMERA SHAKE/i)).toBeInTheDocument()
  })

  it('triggers all 8 chaos skills, precision switches, and sliders from ChaosConsole', () => {
    const onTriggerChaos = vi.fn()
    const onChangeLatency = vi.fn()
    const onChangePrecision = vi.fn()
    const onChangeDenoiseSteps = vi.fn()

    render(
      <ChaosConsole
        activeChaos={[]}
        cooldowns={{}}
        latencyMs={0}
        precision="bf16"
        denoiseSteps={8}
        onTriggerChaos={onTriggerChaos}
        onChangeLatency={onChangeLatency}
        onChangePrecision={onChangePrecision}
        onChangeDenoiseSteps={onChangeDenoiseSteps}
      />,
    )

    fireEvent.click(screen.getByText('推一把'))
    fireEvent.click(screen.getByText('撞机械臂'))
    fireEvent.click(screen.getByText('桌面抹油'))
    fireEvent.click(screen.getByText('灯光突变'))
    fireEvent.click(screen.getByText('镜头抖动'))
    fireEvent.click(screen.getByText('遮挡贴纸'))
    fireEvent.click(screen.getByText(/SPIKE \[7\]/i))
    fireEvent.click(screen.getByText('腕部断流 2s'))

    expect(onTriggerChaos.mock.calls.map((c) => c[0])).toEqual([
      'object_nudge',
      'arm_kick',
      'friction_glitch',
      'lighting_shift',
      'camera_shake',
      'occlusion_patch',
      'latency_inject',
      'wrist_blackout',
    ])

    fireEvent.click(screen.getByText('NVFP4'))
    expect(onChangePrecision).toHaveBeenCalledWith('nvfp4')

    const sliders = screen.getAllByRole('slider')
    fireEvent.change(sliders[0], { target: { value: '180' } })
    expect(onChangeLatency).toHaveBeenCalledWith(180, false)

    fireEvent.change(sliders[1], { target: { value: '3' } })
    expect(onChangeDenoiseSteps).toHaveBeenCalledWith(3)
  })
})

describe('BenchmarkGallery view', () => {
  it('renders 5-dim radar chart, latency comparison, filters, and dual-screen GIF player', async () => {
    render(<BenchmarkGallery />)
    await act(async () => {
      await Promise.resolve()
    })

    expect(screen.getByText(/抗压榜单与破防回放馆/i)).toBeInTheDocument()
    expect(
      screen.getByRole('img', { name: /5-dimensional stress resilience radar chart/i }),
    ).toBeInTheDocument()
    expect(screen.getByText(/LATENCY & THROUGHPUT COMPARISON/i)).toBeInTheDocument()
    expect(screen.getByText(/DUAL-SCREEN GIF REPLAY COMPARISON/i)).toBeInTheDocument()

    // Filter to NVFP4 precision
    fireEvent.click(screen.getByRole('button', { name: 'NVFP4' }))
    expect(screen.getByText('3 / 12')).toBeInTheDocument()

    // Switch replay scenario and trigger sync restart
    fireEvent.click(screen.getByText(/2s Wrist Cam Blackout/i))
    expect(screen.getByText(/CHAOS: wrist_blackout/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /SYNC RESTART/i }))

    // Switch static benchmark dataset
    const selector = screen.getByLabelText('Select benchmark result JSON')
    fireEvent.change(selector, { target: { value: 'cpu_mock_baseline.json' } })
    expect(screen.getByText(/M1 CPU Mock Policy Verification Suite/i)).toBeInTheDocument()
  })
})

describe('App Live WebSocket flow, disconnect/reconnect, and view navigation', () => {
  const originalWs = globalThis.WebSocket

  afterEach(() => {
    vi.useRealTimers()
    MockLiveWebSocket.instances = []
    globalThis.WebSocket = originalWs
    window.history.replaceState(null, '', '/')
  })

  it('streams Live WS frames, sends chaos/config/control messages, auto-reconnects, and switches views', async () => {
    vi.useFakeTimers()
    globalThis.WebSocket = MockLiveWebSocket as unknown as typeof WebSocket

    render(<App />)
    expect(MockLiveWebSocket.instances).toHaveLength(1)
    const ws1 = MockLiveWebSocket.instances[0]

    act(() => {
      ws1.emitOpen()
      ws1.emitJson(makeSampleFrame())
    })

    expect(screen.getByText('LIVE')).toBeInTheDocument()

    // Trigger chaos + precision + pause + export_gif over Live WS
    fireEvent.click(screen.getByText('推一把'))
    fireEvent.click(screen.getByText('FP8'))
    fireEvent.click(screen.getByRole('button', { name: /PAUSE/i }))
    fireEvent.click(screen.getByRole('button', { name: /EXPORT GIF/i }))

    const parsedSent = ws1.sent.map((raw) => JSON.parse(raw) as Record<string, unknown>)
    expect(parsedSent).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'chaos', skill: 'object_nudge' }),
        expect.objectContaining({ type: 'config', precision: 'fp8' }),
        expect.objectContaining({ type: 'control', action: 'pause' }),
        expect.objectContaining({ type: 'control', action: 'export_gif' }),
      ]),
    )

    // Simulate unexpected disconnect -> automatic reconnect
    act(() => {
      ws1.emitUnexpectedClose()
    })
    expect(screen.getByText(/RETRY 1\/3/i)).toBeInTheDocument()

    act(() => {
      vi.advanceTimersByTime(1300)
    })
    expect(MockLiveWebSocket.instances).toHaveLength(2)

    const ws2 = MockLiveWebSocket.instances[1]
    act(() => {
      ws2.emitOpen()
      ws2.emitJson(makeSampleFrame({ step: 0 }))
    })
    expect(screen.getByText('LIVE')).toBeInTheDocument()

    // Navigate to Benchmark & Replay Gallery view
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /BENCHMARK & REPLAY/i }))
      await Promise.resolve()
    })
    expect(screen.getByText(/抗压榜单与破防回放馆/i)).toBeInTheDocument()
    expect(window.location.hash).toBe('#gallery')
  })
})
