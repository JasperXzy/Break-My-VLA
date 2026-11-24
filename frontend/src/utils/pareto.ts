import type { BenchmarkEntry } from '../types/benchmark'

/**
 * Computes the Pareto-optimal frontier minimizing inference latency (mean_ms)
 * and maximizing composite resilience score (overall_score).
 * Returned entries are sorted by latency.mean_ms ascending.
 */
export function computeParetoFrontier(entries: BenchmarkEntry[]): BenchmarkEntry[] {
  if (entries.length === 0) return []

  const sorted = [...entries].sort((a, b) => {
    const latDiff = a.latency.mean_ms - b.latency.mean_ms
    if (Math.abs(latDiff) > 1e-6) return latDiff
    return b.overall_score - a.overall_score
  })

  const frontier: BenchmarkEntry[] = []
  let bestScore = -Infinity

  for (const entry of sorted) {
    if (entry.overall_score > bestScore + 1e-6) {
      frontier.push(entry)
      bestScore = entry.overall_score
    } else if (
      frontier.length > 0 &&
      Math.abs(entry.latency.mean_ms - frontier[frontier.length - 1].latency.mean_ms) <= 1e-6 &&
      Math.abs(entry.overall_score - bestScore) <= 1e-6
    ) {
      // Exact tie on both objectives is also non-dominated
      frontier.push(entry)
    }
  }

  return frontier
}
