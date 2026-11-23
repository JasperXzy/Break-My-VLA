import fs from 'node:fs'
import path from 'node:path'
import type { Plugin } from 'vite'

const VIRTUAL_ID = 'virtual:benchmark-results'
const RESOLVED_VIRTUAL_ID = '\0' + VIRTUAL_ID

interface RawBundledBenchmark {
  fileName: string
  sourcePath: string
  origin: 'repo' | 'public'
  raw: unknown
}

function readJsonDir(dirPath: string, origin: 'repo' | 'public'): RawBundledBenchmark[] {
  if (!fs.existsSync(dirPath)) return []
  const entries = fs.readdirSync(dirPath).filter((f) => f.endsWith('.json') && f !== 'index.json')
  const items: RawBundledBenchmark[] = []
  for (const fileName of entries.sort()) {
    const fullPath = path.join(dirPath, fileName)
    try {
      const content = fs.readFileSync(fullPath, 'utf-8')
      items.push({
        fileName,
        sourcePath: `benchmarks/results/${fileName}`,
        origin,
        raw: JSON.parse(content),
      })
    } catch {
      // Skip malformed JSON files instead of breaking build
    }
  }
  return items
}

/**
 * Exposes static benchmark JSON files from both `<repo>/benchmarks/results/*.json`
 * (higher priority) and `frontend/public/benchmarks/results/*.json` via
 * `virtual:benchmark-results`, and mirrors repo JSONs into `dist/benchmarks/results/`
 * for GitHub Pages static hosting.
 */
export function benchmarkResultsPlugin(frontendRoot: string): Plugin {
  const repoBenchDir = path.resolve(frontendRoot, '..', 'benchmarks', 'results')
  const publicBenchDir = path.resolve(frontendRoot, 'public', 'benchmarks', 'results')

  const collectAll = (): RawBundledBenchmark[] => {
    const byFile = new Map<string, RawBundledBenchmark>()
    for (const item of readJsonDir(publicBenchDir, 'public')) {
      byFile.set(item.fileName, item)
    }
    for (const item of readJsonDir(repoBenchDir, 'repo')) {
      byFile.set(item.fileName, item)
    }
    return Array.from(byFile.values())
  }

  return {
    name: 'break-my-vla-benchmark-results',
    resolveId(id) {
      if (id === VIRTUAL_ID) return RESOLVED_VIRTUAL_ID
      return null
    },
    load(id) {
      if (id !== RESOLVED_VIRTUAL_ID) return null
      const datasets = collectAll()
      return `export const BUNDLED_BENCHMARKS = ${JSON.stringify(datasets)};`
    },
    writeBundle(options) {
      const outDir = options.dir
        ? path.resolve(options.dir)
        : path.resolve(frontendRoot, 'dist')
      const targetDir = path.join(outDir, 'benchmarks', 'results')
      const datasets = collectAll()
      if (datasets.length === 0) return

      fs.mkdirSync(targetDir, { recursive: true })
      const manifestFiles: string[] = []
      for (const item of datasets) {
        manifestFiles.push(item.fileName)
        if (item.origin === 'repo') {
          fs.writeFileSync(
            path.join(targetDir, item.fileName),
            JSON.stringify(item.raw, null, 2),
            'utf-8',
          )
        }
      }
      fs.writeFileSync(
        path.join(targetDir, 'index.json'),
        JSON.stringify({ files: manifestFiles }, null, 2),
        'utf-8',
      )
    },
  }
}
