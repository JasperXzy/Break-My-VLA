import { BUNDLED_BENCHMARKS } from 'virtual:benchmark-results'
import { type BenchmarkDataset, normalizeBenchmarkDataset } from '../types/benchmark'

export function getBundledBenchmarks(): BenchmarkDataset[] {
  return BUNDLED_BENCHMARKS.map((item) =>
    normalizeBenchmarkDataset(item.raw, item.fileName, item.sourcePath),
  )
}

/**
 * Fetches static JSON files listed in `./benchmarks/results/index.json` when served
 * over HTTP (e.g. GitHub Pages or Vite preview) and merges with build-time bundled datasets.
 */
export async function loadStaticBenchmarks(): Promise<BenchmarkDataset[]> {
  const map = new Map<string, BenchmarkDataset>()
  for (const ds of getBundledBenchmarks()) {
    map.set(ds.fileName, ds)
  }

  if (typeof window === 'undefined' || typeof fetch === 'undefined') {
    return Array.from(map.values())
  }

  try {
    const manifestRes = await fetch('./benchmarks/results/index.json')
    if (!manifestRes.ok) return Array.from(map.values())
    const manifest = (await manifestRes.json()) as { files?: string[] }
    const files = Array.isArray(manifest.files) ? manifest.files : []

    await Promise.all(
      files.map(async (fileName) => {
        if (!fileName.endsWith('.json')) return
        try {
          const res = await fetch(`./benchmarks/results/${encodeURIComponent(fileName)}`)
          if (!res.ok) return
          const raw = (await res.json()) as unknown
          const ds = normalizeBenchmarkDataset(raw, fileName, `benchmarks/results/${fileName}`)
          map.set(fileName, ds)
        } catch {
          // Keep bundled fallback
        }
      }),
    )
  } catch {
    // Offline or non-HTTP environment; bundled datasets remain available
  }

  return Array.from(map.values())
}

export function parseBenchmarkJsonText(text: string, fileName: string): BenchmarkDataset {
  const parsed = JSON.parse(text) as unknown
  const dataset = normalizeBenchmarkDataset(parsed, fileName, `uploaded/${fileName}`)
  if (dataset.entries.length === 0) {
    throw new Error('JSON must contain a non-empty "entries" array.')
  }
  return dataset
}
