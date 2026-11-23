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

describe('Multi-VLA Head-to-Head Arena (BenchmarkGallery)', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('defaults to real GPU suite with REAL GPU VERIFIED badge, 1v1 Head-to-Head card, Chaos Survival Matrix, and Markdown export', async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    Object.defineProperty(navigator, 'clipboard', {
      configurable: true,
      value: { writeText },
    })

    render(<BenchmarkGallery />)
    await act(async () => {
      await Promise.resolve()
    })

    // Defaults to 01_smolvla_libero_l4_real.json with prominent REAL GPU VERIFIED badge
    expect(screen.getByText(/Multi-VLA Head-to-Head Arena/i)).toBeInTheDocument()
    expect(screen.getByTestId('dataset-verification-badge')).toHaveTextContent('REAL GPU VERIFIED')
    expect(screen.getByText(/M2 Real LIBERO Object · SmolVLA \(NVIDIA L4\)/i)).toBeInTheDocument()

    // Head-to-Head 1v1 Card metrics (BF16 vs INT8 2-step)
    expect(screen.getByText(/HEAD-TO-HEAD 1V1 ARENA/i)).toBeInTheDocument()
    expect(screen.getByTestId('h2h-speedup')).toHaveTextContent('3.17×')
    expect(screen.getByTestId('h2h-vram-savings')).toHaveTextContent('-0.10 GB')
    expect(screen.getByTestId('h2h-overall-delta')).toHaveTextContent('+15.3 pts')
    expect(screen.getByTestId('h2h-verdict')).toHaveTextContent(/3\.17× faster/i)

    // Chaos Survival Matrix heatmap cells
    expect(screen.getByText(/CHAOS SURVIVAL MATRIX/i)).toBeInTheDocument()
    expect(
      screen.getByTestId('heatmap-cell-smolvla-libero-l4-bf16-physical'),
    ).toHaveTextContent('0%')
    expect(
      screen.getByTestId('heatmap-cell-smolvla-libero-l4-int8-2step-systems'),
    ).toHaveTextContent('95%')
    expect(screen.getByText(/MOST LETHAL: PHYSICAL/i)).toBeInTheDocument()

    // One-click Copy Markdown & Export .md
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /COPY MARKDOWN/i }))
      await Promise.resolve()
    })
    expect(writeText).toHaveBeenCalledTimes(1)
    const copiedMd = writeText.mock.calls[0][0] as string
    expect(copiedMd).toContain('REAL GPU VERIFIED')
    expect(copiedMd).toContain('| Metric |')
    expect(copiedMd).toContain('3.17× faster')
    expect(screen.getByText(/COPIED MD!/i)).toBeInTheDocument()

    const anchorClickSpy = vi
      .spyOn(HTMLAnchorElement.prototype, 'click')
      .mockImplementation(() => {})
    fireEvent.click(screen.getByRole('button', { name: /EXPORT \.MD/i }))
    expect(URL.createObjectURL).toHaveBeenCalled()
    expect(anchorClickSpy).toHaveBeenCalledTimes(1)

    // Switch to Showcase dataset -> badge changes & supports 1v1 contender picking, filtering, and replay switching
    const selector = screen.getByLabelText('Select benchmark result JSON')
    fireEvent.change(selector, { target: { value: 'auto_chaos_m4_showcase.json' } })
    expect(screen.getByTestId('dataset-verification-badge')).toHaveTextContent(
      'SHOWCASE / REFERENCE',
    )

    // Pick arbitrary 2 contenders in 1v1 Head-to-Head card
    const selectA = screen.getByLabelText('Select Contender A')
    const selectB = screen.getByLabelText('Select Contender B')
    fireEvent.change(selectA, { target: { value: 'pi05-bf16' } })
    fireEvent.change(selectB, { target: { value: 'openvla-nvfp4' } })
    expect(screen.getByTestId('h2h-speedup')).toHaveTextContent('2.54×')
    expect(screen.getByTestId('h2h-vram-savings')).toHaveTextContent('-6.10 GB')

    // Swap A <-> B
    fireEvent.click(screen.getByRole('button', { name: /SWAP A↔B/i }))
    expect((selectA as HTMLSelectElement).value).toBe('openvla-nvfp4')
    expect((selectB as HTMLSelectElement).value).toBe('pi05-bf16')

    // Filter to NVFP4 precision
    fireEvent.click(screen.getByRole('button', { name: 'NVFP4' }))
    expect(screen.getByText('3 / 12')).toBeInTheDocument()

    // Switch replay scenario and trigger sync restart
    expect(
      screen.getByRole('img', { name: /5-dimensional stress resilience radar chart/i }),
    ).toBeInTheDocument()
    expect(screen.getByText(/LATENCY & THROUGHPUT COMPARISON/i)).toBeInTheDocument()
    expect(screen.getByText(/DUAL-SCREEN GIF REPLAY COMPARISON/i)).toBeInTheDocument()
    fireEvent.click(screen.getByText(/2s Wrist Cam Blackout/i))
    expect(screen.getByText(/CHAOS: wrist_blackout/i)).toBeInTheDocument()
    fireEvent.click(screen.getByRole('button', { name: /SYNC RESTART/i }))

    // Switch to CPU Mock dataset -> badge shows CPU MOCK BASELINE
    fireEvent.change(selector, { target: { value: 'cpu_mock_baseline.json' } })
    expect(screen.getByTestId('dataset-verification-badge')).toHaveTextContent('CPU MOCK BASELINE')
    expect(screen.getByText(/M1 CPU Mock Policy Verification Suite/i)).toBeInTheDocument()
  })
})

describe('App Live WebSocket flow, multi-model switcher, disconnect/reconnect, and view navigation', () => {
  const originalWs = globalThis.WebSocket
  const originalFetch = globalThis.fetch

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
    MockLiveWebSocket.instances = []
    globalThis.WebSocket = originalWs
    globalThis.fetch = originalFetch
    window.history.replaceState(null, '', '/')
  })

  it('loads multi-model /health, sends {type:"config", model} on switch, handles reconnect, and navigates to Arena', async () => {
    vi.useFakeTimers()
    globalThis.WebSocket = MockLiveWebSocket as unknown as typeof WebSocket
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        active_model: 'pi0.5-libero',
        models: [
          {
            id: 'smolvla-450m',
            label: 'SmolVLA 450M',
            params_b: 0.45,
            baseline_latency_ms: 24,
          },
          {
            id: 'pi0.5-libero',
            label: 'Pi0.5',
            params_b: 3.0,
            baseline_latency_ms: 58,
          },
          {
            id: 'isaac-gr00t-n1',
            label: 'Isaac GR00T',
            params_b: 2.2,
            baseline_latency_ms: 42,
          },
        ],
        device: 'cuda:0',
      }),
    } as Response)

    render(<App />)
    expect(MockLiveWebSocket.instances).toHaveLength(1)
    const ws1 = MockLiveWebSocket.instances[0]

    await act(async () => {
      ws1.emitOpen()
      ws1.emitJson(makeSampleFrame())
      await Promise.resolve()
      await Promise.resolve()
    })

    expect(screen.getByText('LIVE')).toBeInTheDocument()
    const modelSelect = screen.getByLabelText('Select active VLA model') as HTMLSelectElement
    expect(modelSelect.value).toBe('pi0.5-libero')
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('3.0B')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 58ms')

    // Hot-switch active VLA model over WebSocket
    fireEvent.change(modelSelect, { target: { value: 'isaac-gr00t-n1' } })
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('2.2B')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 42ms')

    // Trigger chaos + precision + pause + export_gif over Live WS
    fireEvent.click(screen.getByText('推一把'))
    fireEvent.click(screen.getByText('FP8'))
    fireEvent.click(screen.getByRole('button', { name: /PAUSE/i }))
    fireEvent.click(screen.getByRole('button', { name: /EXPORT GIF/i }))

    const parsedSent = ws1.sent.map((raw) => JSON.parse(raw) as Record<string, unknown>)
    expect(parsedSent).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'config', model: 'isaac-gr00t-n1' }),
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
    await act(async () => {
      ws2.emitOpen()
      ws2.emitJson(makeSampleFrame({ step: 0 }))
      await Promise.resolve()
    })
    expect(screen.getByText('LIVE')).toBeInTheDocument()

    // Navigate to Multi-VLA Arena view
    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /MULTI-VLA ARENA/i }))
      await Promise.resolve()
    })
    expect(screen.getByText(/Multi-VLA Head-to-Head Arena/i)).toBeInTheDocument()
    expect(window.location.hash).toBe('#gallery')
  })

  it('remains backward-compatible with legacy single-model /health responses', async () => {
    globalThis.WebSocket = MockLiveWebSocket as unknown as typeof WebSocket
    globalThis.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => ({
        model: 'lerobot/smolvla_libero',
        precisions: ['bf16', 'int8'],
        device: 'cuda:0',
        env: 'LiberoEnvAdapter',
      }),
    } as Response)

    render(<App />)
    const ws = MockLiveWebSocket.instances[0]
    await act(async () => {
      ws.emitOpen()
      ws.emitJson(makeSampleFrame())
      await Promise.resolve()
      await Promise.resolve()
    })

    const modelSelect = screen.getByLabelText('Select active VLA model') as HTMLSelectElement
    expect(modelSelect.value).toBe('lerobot/smolvla_libero')
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('450M')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 24ms')
  })

  it('provides 4 differentiated Demo model presets and updates params & latency baseline live', () => {
    vi.useFakeTimers()
    window.history.replaceState(null, '', '/?mode=demo')

    render(<App />)

    const modelSelect = screen.getByLabelText('Select active VLA model') as HTMLSelectElement
    expect(Array.from(modelSelect.options)).toHaveLength(4)
    expect(modelSelect.value).toBe('smolvla-450m')
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('450M')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 24ms')

    fireEvent.change(modelSelect, { target: { value: 'openvla-oft-7b' } })
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('7.0B')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 74ms')

    fireEvent.change(modelSelect, { target: { value: 'isaac-gr00t-n1' } })
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('2.2B')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 42ms')

    fireEvent.change(modelSelect, { target: { value: 'pi0.5-libero' } })
    expect(screen.getByTestId('active-model-params')).toHaveTextContent('3.0B')
    expect(screen.getByTestId('active-model-baseline')).toHaveTextContent('BASE 58ms')
  })
})
