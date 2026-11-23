import { describe, expect, it } from 'vitest'
import { getBundledBenchmarks, parseBenchmarkJsonText } from '../data/benchmarkLoader'
import { buildDefaultChaosParams } from '../hooks/useVlaSession'
import { normalizeBenchmarkDataset } from '../types/benchmark'
import {
  CHAOS_SKILL_META,
  type ChaosSkill,
  toImageDataUri,
} from '../types/protocol'
import { encodeFramesToGifBlob } from '../utils/gifEncoder'

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

describe('Benchmark JSON loader and normalization', () => {
  it('loads bundled static benchmark datasets from benchmarks/results', () => {
    const datasets = getBundledBenchmarks()
    expect(datasets.length).toBeGreaterThanOrEqual(2)
    const mainSuite = datasets.find((d) => d.fileName === 'auto_chaos_m4_showcase.json')
    expect(mainSuite).toBeDefined()
    expect(mainSuite?.entries.length).toBeGreaterThanOrEqual(8)
    expect(mainSuite?.replays.length).toBeGreaterThanOrEqual(3)
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
})
