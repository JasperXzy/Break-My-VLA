import { describe, expect, it } from 'vitest'
import {
  getBundledBenchmarks,
  hasM4Telemetry,
  parseBenchmarkJsonText,
} from '../data/benchmarkLoader'
import { buildDefaultChaosParams } from '../hooks/useVlaSession'
import { DemoSimulationEngine } from '../sim/demoEngine'
import { classifyBenchmarkDataset, normalizeBenchmarkDataset } from '../types/benchmark'
import {
  CHAOS_SKILL_META,
  type ChaosSkill,
  DEMO_MODEL_PRESETS,
  parseHealthModels,
  toImageDataUri,
} from '../types/protocol'
import { buildArenaShareHash, parseArenaHash } from '../utils/arenaHash'
import {
  computeHeadToHeadMetrics,
  formatHeadToHeadMarkdown,
} from '../utils/arenaMarkdown'
import { encodeFramesToGifBlob } from '../utils/gifEncoder'
import { computeParetoFrontier } from '../utils/pareto'

describe('Wire protocol contract helpers', () => {
  it('generates valid schema-bounded default params for all 8 chaos skills', () => {
    const allSkills = Object.keys(CHAOS_SKILL_META) as ChaosSkill[]
    expect(allSkills).toHaveLength(8)

    const nudge = buildDefaultChaosParams('object_nudge')
    expect(Math.abs(nudge.dx)).toBeLessThanOrEqual(0.5)
    expect(Math.abs(nudge.dy)).toBeLessThanOrEqual(0.5)
    expect(Math.abs(nudge.dz)).toBeLessThanOrEqual(0.5)

    const kick = buildDefaultChaosParams('arm_kick')
    expect(Math.abs(kick.dx)).toBeLessThanOrEqual(0.5)
    expect(Math.abs(kick.dy)).toBeLessThanOrEqual(0.5)
    expect(Math.abs(kick.dz)).toBeLessThanOrEqual(0.5)

    const friction = buildDefaultChaosParams('friction_glitch')
    expect(Number.isInteger(friction.duration_steps)).toBe(true)
    expect(friction.multiplier).toBeGreaterThanOrEqual(0)
    expect(friction.multiplier).toBeLessThanOrEqual(3)

    const lighting = buildDefaultChaosParams('lighting_shift')
    expect(Number.isInteger(lighting.duration_steps)).toBe(true)
    expect(lighting.brightness).toBeGreaterThanOrEqual(0)
    expect(lighting.brightness).toBeLessThanOrEqual(2.5)
    expect(lighting.saturation).toBeGreaterThanOrEqual(0)
    expect(lighting.saturation).toBeLessThanOrEqual(2.5)

    const shake = buildDefaultChaosParams('camera_shake')
    expect(Number.isInteger(shake.duration_steps)).toBe(true)
    expect(Number.isInteger(shake.amplitude_px)).toBe(true)
    expect(shake.amplitude_px).toBeGreaterThanOrEqual(0)
    expect(shake.amplitude_px).toBeLessThanOrEqual(80)

    const occ = buildDefaultChaosParams('occlusion_patch')
    expect(occ.x).toBeGreaterThanOrEqual(0)
    expect(occ.x).toBeLessThan(1)
    expect(occ.y).toBeGreaterThanOrEqual(0)
    expect(occ.y).toBeLessThan(1)
    expect(occ.width).toBeGreaterThan(0)
    expect(occ.width).toBeLessThanOrEqual(1)
    expect(occ.height).toBeGreaterThan(0)
    expect(occ.height).toBeLessThanOrEqual(1)

    const latency = buildDefaultChaosParams('latency_inject', 220)
    expect(latency.latency_ms).toBe(220)
    expect(latency.latency_ms).toBeLessThanOrEqual(300)

    const blackout = buildDefaultChaosParams('wrist_blackout')
    expect(Number.isInteger(blackout.duration_steps)).toBe(true)
    expect(blackout.duration_steps).toBeGreaterThanOrEqual(1)
    expect(blackout.duration_steps).toBeLessThanOrEqual(600)
  })

  it('normalizes raw base64 and data URIs for camera frames', () => {
    expect(toImageDataUri('/9j/abc')).toBe('data:image/jpeg;base64,/9j/abc')
    expect(toImageDataUri('iVBORxyz')).toBe('data:image/png;base64,iVBORxyz')
    expect(toImageDataUri('data:image/jpeg;base64,already')).toBe(
      'data:image/jpeg;base64,already',
    )
    expect(toImageDataUri('')).toBe('')
  })

  it('encodes valid GIF89a binary output in local demo encoder', async () => {
    const frameA = {
      width: 4,
      height: 4,
      data: new Uint8ClampedArray(4 * 4 * 4).fill(120),
    } as ImageData
    const frameB = {
      width: 4,
      height: 4,
      data: new Uint8ClampedArray(4 * 4 * 4).fill(240),
    } as ImageData

    const blob = encodeFramesToGifBlob([frameA, frameB], 8)
    expect(blob.type).toBe('image/gif')
    const buffer = await new Promise<ArrayBuffer>((resolve, reject) => {
      const reader = new FileReader()
      reader.onload = () => resolve(reader.result as ArrayBuffer)
      reader.onerror = () => reject(reader.error)
      reader.readAsArrayBuffer(blob)
    })
    const bytes = new Uint8Array(buffer)
    const header = String.fromCharCode(...bytes.slice(0, 6))
    expect(header).toBe('GIF89a')
    expect(bytes[bytes.length - 1]).toBe(0x3b)
  })
})

describe('Multi-VLA /health parsing and Demo presets', () => {
  it('supports backward-compatible single-model /health responses', () => {
    const parsed = parseHealthModels({
      model: 'lerobot/smolvla_libero',
      precisions: ['bf16', 'int8'],
      device: 'cuda:0',
      env: 'LiberoEnvAdapter',
    })
    expect(parsed).not.toBeNull()
    expect(parsed?.activeModel).toBe('lerobot/smolvla_libero')
    expect(parsed?.models).toHaveLength(1)
    expect(parsed?.models[0].paramsLabel).toBe('450M')
    expect(parsed?.models[0].baselineLatencyMs).toBe(24)
    expect(parsed?.models[0].precisions).toEqual(['bf16', 'int8'])
  })

  it('parses multi-model /health with string list and active_model', () => {
    const parsed = parseHealthModels({
      active_model: 'pi0.5-libero',
      models: ['smolvla-450m', 'pi0.5-libero', 'isaac-gr00t-n1', 'openvla-oft-7b'],
      device: 'cuda:0',
    })
    expect(parsed).not.toBeNull()
    expect(parsed?.activeModel).toBe('pi0.5-libero')
    expect(parsed?.models.map((m) => m.id)).toEqual([
      'smolvla-450m',
      'pi0.5-libero',
      'isaac-gr00t-n1',
      'openvla-oft-7b',
    ])
    expect(parsed?.models.map((m) => m.paramsLabel)).toEqual(['450M', '3.0B', '2.2B', '7.0B'])
  })

  it('parses multi-model /health with rich model metadata objects', () => {
    const parsed = parseHealthModels({
      active_model: 'groot-custom',
      models: [
        {
          id: 'groot-custom',
          label: 'GR00T N1.5 Custom',
          params_b: 2.5,
          baseline_latency_ms: 39,
          precisions: ['bf16', 'fp8'],
        },
        {
          id: 'smolvla-l4',
          label: 'SmolVLA L4',
          params: '450M',
          baseline_latency_ms: 120,
        },
      ],
    })
    expect(parsed?.activeModel).toBe('groot-custom')
    expect(parsed?.models[0]).toMatchObject({
      id: 'groot-custom',
      label: 'GR00T N1.5 Custom',
      paramsLabel: '2.5B',
      baselineLatencyMs: 39,
      precisions: ['bf16', 'fp8'],
    })
    expect(parsed?.models[1]).toMatchObject({
      id: 'smolvla-l4',
      label: 'SmolVLA L4',
      paramsLabel: '450M',
      baselineLatencyMs: 120,
    })
  })

  it('provides 4 differentiated Demo presets with distinct latency profiles in DemoSimulationEngine', () => {
    expect(DEMO_MODEL_PRESETS.map((p) => p.label)).toEqual([
      'SmolVLA 450M',
      'Pi0.5',
      'Isaac GR00T',
      'OpenVLA-OFT',
    ])

    const engine = new DemoSimulationEngine()
    engine.configure({ model: 'smolvla-450m', precision: 'bf16', denoise_steps: 8 })
    const smolFrame = engine.tick()
    expect(smolFrame.model).toBe('smolvla-450m')

    engine.configure({ model: 'openvla-oft-7b', precision: 'bf16', denoise_steps: 8 })
    const openvlaFrame = engine.tick()
    expect(openvlaFrame.model).toBe('openvla-oft-7b')

    expect(openvlaFrame.infer_ms).toBeGreaterThan(smolFrame.infer_ms + 30)
  })
})

describe('Benchmark JSON loader, provenance sorting, and Head-to-Head Markdown export', () => {
  it('prioritizes real GPU verified datasets first when loading bundled benchmarks', () => {
    const datasets = getBundledBenchmarks()
    expect(datasets.length).toBeGreaterThanOrEqual(3)

    // Real GPU dataset is prioritized at index 0
    expect(datasets[0].fileName).toBe('01_smolvla_libero_l4_real.json')
    expect(datasets[0].verification).toBe('real_gpu')

    const showcase = datasets.find((d) => d.fileName === 'auto_chaos_m4_showcase.json')
    expect(showcase?.verification).toBe('showcase')
    expect(showcase?.entries.length).toBeGreaterThanOrEqual(8)

    const cpuMock = datasets.find((d) => d.fileName === 'cpu_mock_baseline.json')
    expect(cpuMock?.verification).toBe('cpu_mock')
  })

  it('classifies real GPU vs showcase vs CPU mock provenance accurately', () => {
    expect(
      classifyBenchmarkDataset({
        fileName: '01_smolvla_libero_l4_real.json',
        suite_name: 'M2 Real LIBERO Object',
        hardware: 'NVIDIA L4 (24 GB)',
        notes: 'Measured closed-loop run',
      }),
    ).toBe('real_gpu')
    expect(
      classifyBenchmarkDataset({
        fileName: 'auto_chaos_m4_showcase.json',
        suite_name: 'Showcase Matrix',
        hardware: 'NVIDIA Blackwell Preview',
        notes: 'Static reference benchmark',
      }),
    ).toBe('showcase')
    expect(
      classifyBenchmarkDataset({
        fileName: 'cpu_mock_baseline.json',
        suite_name: 'M1 CPU Mock Policy',
        hardware: 'gcp-dev 4 vCPU (Mock Policy Server)',
        notes: '',
      }),
    ).toBe('cpu_mock')
  })

  it('computes 1v1 Head-to-Head deltas, speedup ratio, VRAM savings, and valid GFM Markdown table', () => {
    const datasets = getBundledBenchmarks()
    const realSuite = datasets[0]
    expect(realSuite.entries.length).toBeGreaterThanOrEqual(2)

    const [bf16, int8] = realSuite.entries
    const metrics = computeHeadToHeadMetrics(bf16, int8)
    expect(metrics.overallDelta).toBe(15.3)
    expect(metrics.overallWinner).toBe('b')
    expect(metrics.fasterSide).toBe('b')
    expect(metrics.speedupRatio).toBeCloseTo(3.17, 1)
    expect(metrics.p50SpeedupRatio).toBeCloseTo(2.87, 1)
    expect(metrics.vramWinner).toBe('b')
    expect(metrics.vramSavedGb).toBeCloseTo(0.1, 2)
    expect(metrics.axisDeltas).toHaveLength(5)
    expect(metrics.axisDeltas.find((a) => a.key === 'systems')?.delta).toBe(27)

    const md = formatHeadToHeadMarkdown(realSuite, bf16, int8)
    expect(md).toContain('REAL GPU VERIFIED')
    expect(md).toContain('| Metric |')
    expect(md).toContain('Overall Resilience')
    expect(md).toContain('3.17× faster')
    expect(md).toContain('Peak VRAM')
    expect(md).toContain('Verdict')
  })

  it('normalizes ratio scores (0..1) and uppercase precision names', () => {
    const ds = normalizeBenchmarkDataset(
      {
        suite_name: 'Ratio Suite',
        entries: [
          {
            model: 'pi05',
            precision: 'NVFP4',
            denoise_steps: 2,
            infer_ms: 16.5,
            radar_scores: {
              clean: 0.9,
              physical: 0.7,
              visual: 0.65,
              systems: 0.8,
              quantization: 0.55,
            },
          },
        ],
      },
      'ratio.json',
      'benchmarks/results/ratio.json',
    )

    expect(ds.entries).toHaveLength(1)
    expect(ds.entries[0].precision).toBe('nvfp4')
    expect(ds.entries[0].radar_scores.clean).toBe(90)
    expect(ds.entries[0].overall_score).toBe(72)
  })

  it('rejects uploaded JSON without entries', () => {
    expect(() => parseBenchmarkJsonText('{"suite_name":"empty"}', 'bad.json')).toThrow(
      /entries/i,
    )
  })

  it('normalizes M4 TensorRT backend, p99_ms, and BF16 alignment metrics while keeping legacy entries backward compatible', () => {
    const ds = normalizeBenchmarkDataset(
      {
        suite_name: 'M4 TRT & Legacy Mixed Suite',
        entries: [
          {
            id: 'legacy-bf16',
            model: 'SmolVLA',
            precision: 'bf16',
            denoise_steps: 10,
            latency: { mean_ms: 42.0, p50_ms: 41.0, p95_ms: 49.0, fps: 23.8 },
            radar_scores: { clean: 90, physical: 70, visual: 80, systems: 75, quantization: 85 },
          },
          {
            id: 'groot-nvfp4',
            model: 'GR00T-N1.5',
            backend: 'TRT',
            precision: 'nvfp4',
            denoise_steps: 4,
            latency: { mean_ms: 12.6, p50_ms: 12.3, p95_ms: 14.4, p99_ms: 15.5, fps: 79.4 },
            radar_scores: { clean: 91, physical: 77, visual: 75, systems: 78, quantization: 67 },
            alignment: {
              action_mse_vs_bf16: 0.00018,
              action_l1_vs_bf16: 0.0085,
              action_cos_vs_bf16: 0.9994,
            },
          },
        ],
      },
      'm4_mixed.json',
      'benchmarks/results/m4_mixed.json',
    )

    const [legacy, trt] = ds.entries
    expect(legacy.backend).toBeUndefined()
    expect(legacy.latency.p99_ms).toBeUndefined()
    expect(legacy.alignment).toBeUndefined()

    expect(trt.backend).toBe('tensorrt')
    expect(trt.latency.p99_ms).toBe(15.5)
    expect(trt.alignment).toEqual({
      action_mse_vs_bf16: 0.00018,
      action_l1_vs_bf16: 0.0085,
      action_cos_vs_bf16: 0.9994,
    })
    expect(hasM4Telemetry(ds)).toBe(true)

    const bundled = getBundledBenchmarks()
    expect(hasM4Telemetry(bundled[0])).toBe(false)
    const showcase = bundled.find((d) => d.fileName === 'auto_chaos_m4_showcase.json')
    expect(showcase).toBeDefined()
    expect(hasM4Telemetry(showcase!)).toBe(true)
    expect(showcase!.entries.some((e) => e.backend === 'tensorrt')).toBe(true)
    expect(showcase!.entries.some((e) => e.model.includes('GR00T'))).toBe(true)
  })

  it('computes the latency-vs-resilience Pareto frontier accurately', () => {
    expect(computeParetoFrontier([])).toEqual([])

    const showcase = getBundledBenchmarks().find(
      (d) => d.fileName === 'auto_chaos_m4_showcase.json',
    )!
    const frontier = computeParetoFrontier(showcase.entries)
    expect(frontier.map((e) => e.id)).toEqual([
      'smolvla-nvfp4',
      'groot-nvfp4-trt',
      'groot-fp8-trt',
      'groot-bf16-trt',
      'groot-bf16',
    ])
    // Frontier must be strictly increasing in overall_score and non-decreasing in mean_ms
    for (let i = 1; i < frontier.length; i++) {
      expect(frontier[i].latency.mean_ms).toBeGreaterThanOrEqual(frontier[i - 1].latency.mean_ms)
      expect(frontier[i].overall_score).toBeGreaterThan(frontier[i - 1].overall_score)
    }
  })

  it('includes Backend, P99 tail latency, and Action Cosine/MSE columns in M4 Markdown export', () => {
    const showcase = getBundledBenchmarks().find(
      (d) => d.fileName === 'auto_chaos_m4_showcase.json',
    )!
    const grootBf16 = showcase.entries.find((e) => e.id === 'groot-bf16')!
    const grootNvfp4 = showcase.entries.find((e) => e.id === 'groot-nvfp4-trt')!

    const metrics = computeHeadToHeadMetrics(grootBf16, grootNvfp4)
    expect(metrics.hasM4Metrics).toBe(true)
    expect(metrics.p99Winner).toBe('b')
    expect(metrics.p99DeltaMs).toBeCloseTo(-39.9, 1)
    expect(metrics.p99SpeedupRatio).toBeCloseTo(3.57, 1)
    expect(metrics.cosDelta).toBeCloseTo(-0.0154, 4)
    expect(metrics.mseDelta).toBeCloseTo(0.00342, 5)

    const md = formatHeadToHeadMarkdown(showcase, grootBf16, grootNvfp4)
    expect(md).toContain('| Side | Model | Backend | Precision |')
    expect(md).toContain('Action Cosine (vs BF16)')
    expect(md).toContain('Action MSE (vs BF16)')
    expect(md).toContain('TensorRT')
    expect(md).toContain('PyTorch')
    expect(md).toContain('P99 Tail Latency')
    expect(md).toContain('0.9846')
    expect(md).toContain('0.00342')
  })

  it('parses and builds #gallery deep-link share hashes cleanly', () => {
    const hash = buildArenaShareHash({
      dataset: 'auto_chaos_m4_showcase.json',
      a: 'groot-bf16',
      b: 'groot-nvfp4-trt',
    })
    expect(hash).toBe(
      '#gallery?dataset=auto_chaos_m4_showcase.json&a=groot-bf16&b=groot-nvfp4-trt',
    )

    const parsed = parseArenaHash(hash)
    expect(parsed).toEqual({
      view: 'gallery',
      hasExplicitView: true,
      dataset: 'auto_chaos_m4_showcase.json',
      a: 'groot-bf16',
      b: 'groot-nvfp4-trt',
    })

    expect(parseArenaHash('#gallery')).toMatchObject({ view: 'gallery', hasExplicitView: true })
    expect(parseArenaHash('#arcade')).toMatchObject({ view: 'arcade', hasExplicitView: true })
    expect(parseArenaHash('')).toMatchObject({ view: 'arcade', hasExplicitView: false })
  })
})
