import { useEffect, useMemo, useRef, useState, type ChangeEvent } from 'react'
import {
  CheckSquare,
  Cpu,
  Database,
  FileJson,
  Filter,
  ShieldCheck,
  Sparkles,
  Square,
  Trophy,
  Upload,
  Zap,
} from 'lucide-react'
import {
  getBundledBenchmarks,
  loadStaticBenchmarks,
  parseBenchmarkJsonText,
} from '../data/benchmarkLoader'
import {
  type BenchmarkDataset,
  type DatasetVerification,
  sortBenchmarkDatasets,
} from '../types/benchmark'
import type { Precision } from '../types/protocol'
import { verificationBadgeLabel } from '../utils/arenaMarkdown'
import { ChaosSurvivalMatrix } from './ChaosSurvivalMatrix'
import { DualGifReplayPlayer } from './DualGifReplayPlayer'
import { HeadToHeadCard } from './HeadToHeadCard'
import { LatencyComparisonChart } from './LatencyComparisonChart'
import { StressRadarChart } from './StressRadarChart'

const PRECISION_FILTERS: Array<{ id: Precision | 'all'; label: string }> = [
  { id: 'all', label: 'ALL' },
  { id: 'bf16', label: 'BF16' },
  { id: 'fp8', label: 'FP8' },
  { id: 'int8', label: 'INT8' },
  { id: 'nvfp4', label: 'NVFP4' },
]

function datasetOptionPrefix(v: DatasetVerification): string {
  if (v === 'real_gpu') return '[REAL GPU]'
  if (v === 'cpu_mock') return '[CPU MOCK]'
  return '[SHOWCASE]'
}

function VerificationBadge({ verification }: { verification: DatasetVerification }) {
  const label = verificationBadgeLabel(verification)
  if (verification === 'real_gpu') {
    return (
      <span
        data-testid="dataset-verification-badge"
        className="inline-flex items-center gap-1 rounded-md border border-emerald-400/70 bg-emerald-500/20 px-2 py-0.5 font-mono text-[10px] font-extrabold tracking-wider text-emerald-200 shadow-[0_0_16px_rgba(16,185,129,0.35)]"
      >
        <ShieldCheck className="h-3.5 w-3.5 text-emerald-300" />
        <span>{label}</span>
      </span>
    )
  }
  if (verification === 'cpu_mock') {
    return (
      <span
        data-testid="dataset-verification-badge"
        className="inline-flex items-center gap-1 rounded-md border border-amber-400/50 bg-amber-500/15 px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider text-amber-200"
      >
        <Cpu className="h-3.5 w-3.5 text-amber-300" />
        <span>{label}</span>
      </span>
    )
  }
  return (
    <span
      data-testid="dataset-verification-badge"
      className="inline-flex items-center gap-1 rounded-md border border-fuchsia-400/50 bg-fuchsia-500/15 px-2 py-0.5 font-mono text-[10px] font-bold tracking-wider text-fuchsia-200"
    >
      <Sparkles className="h-3.5 w-3.5 text-fuchsia-300" />
      <span>{label}</span>
    </span>
  )
}

export function BenchmarkGallery() {
  const [datasets, setDatasets] = useState<BenchmarkDataset[]>(() => getBundledBenchmarks())
  const [activeFile, setActiveFile] = useState<string>(
    () => getBundledBenchmarks()[0]?.fileName ?? '',
  )
  const [precisionFilter, setPrecisionFilter] = useState<Precision | 'all'>('all')
  const [modelFilter, setModelFilter] = useState<string>('all')
  const [selectedIds, setSelectedIds] = useState<string[]>([])
  const [contenderAId, setContenderAId] = useState<string>('')
  const [contenderBId, setContenderBId] = useState<string>('')
  const [uploadError, setUploadError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement | null>(null)

  useEffect(() => {
    let active = true
    void loadStaticBenchmarks().then((loaded) => {
      if (!active || loaded.length === 0) return
      setDatasets(loaded)
      setActiveFile((prev) =>
        loaded.some((d) => d.fileName === prev) ? prev : loaded[0].fileName,
      )
    })
    return () => {
      active = false
    }
  }, [])

  const activeDataset = useMemo(
    () => datasets.find((d) => d.fileName === activeFile) ?? datasets[0] ?? null,
    [activeFile, datasets],
  )

  const uniqueModels = useMemo(() => {
    if (!activeDataset) return []
    return Array.from(new Set(activeDataset.entries.map((e) => e.model)))
  }, [activeDataset])

  const filteredEntries = useMemo(() => {
    if (!activeDataset) return []
    return activeDataset.entries.filter((entry) => {
      if (precisionFilter !== 'all' && entry.precision !== precisionFilter) return false
      if (modelFilter !== 'all' && entry.model !== modelFilter) return false
      return true
    })
  }, [activeDataset, modelFilter, precisionFilter])

  // Effective 1v1 Head-to-Head contenders (default to first two entries in dataset)
  const effectiveContenderAId = useMemo(() => {
    if (!activeDataset || activeDataset.entries.length === 0) return ''
    if (activeDataset.entries.some((e) => e.id === contenderAId)) return contenderAId
    return activeDataset.entries[0].id
  }, [activeDataset, contenderAId])

  const effectiveContenderBId = useMemo(() => {
    if (!activeDataset || activeDataset.entries.length === 0) return ''
    if (activeDataset.entries.some((e) => e.id === contenderBId)) return contenderBId
    return (activeDataset.entries[1] ?? activeDataset.entries[0]).id
  }, [activeDataset, contenderBId])

  // Default radar selection: first 4 filtered entries if user hasn't picked custom ones
  const effectiveSelectedIds = useMemo(() => {
    const valid = selectedIds.filter((id) => filteredEntries.some((e) => e.id === id))
    if (valid.length > 0) return valid
    return filteredEntries.slice(0, 4).map((e) => e.id)
  }, [filteredEntries, selectedIds])

  const radarEntries = useMemo(
    () => filteredEntries.filter((e) => effectiveSelectedIds.includes(e.id)),
    [effectiveSelectedIds, filteredEntries],
  )

  const toggleSelectEntry = (id: string) => {
    setSelectedIds((prev) => {
      const current = prev.length > 0 ? prev : effectiveSelectedIds
      if (current.includes(id)) {
        return current.length > 1 ? current.filter((x) => x !== id) : current
      }
      return [...current, id].slice(-4)
    })
  }

  const handleSwapContenders = () => {
    setContenderAId(effectiveContenderBId)
    setContenderBId(effectiveContenderAId)
  }

  const handleFileUpload = async (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return
    try {
      const text = await file.text()
      const parsed = parseBenchmarkJsonText(text, file.name)
      setDatasets((prev) => {
        const rest = prev.filter((d) => d.fileName !== parsed.fileName)
        return sortBenchmarkDatasets([parsed, ...rest])
      })
      setActiveFile(parsed.fileName)
      setModelFilter('all')
      setPrecisionFilter('all')
      setSelectedIds([])
      setContenderAId('')
      setContenderBId('')
      setUploadError(null)
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : 'Invalid benchmark JSON file.')
    } finally {
      e.target.value = ''
    }
  }

  const topScoreEntry = useMemo(() => {
    if (!activeDataset || activeDataset.entries.length === 0) return null
    return [...activeDataset.entries].sort((a, b) => b.overall_score - a.overall_score)[0]
  }, [activeDataset])

  const fastestEntry = useMemo(() => {
    if (!activeDataset || activeDataset.entries.length === 0) return null
    return [...activeDataset.entries].sort((a, b) => a.latency.mean_ms - b.latency.mean_ms)[0]
  }, [activeDataset])

  if (!activeDataset) {
    return (
      <main className="mx-auto w-full max-w-[1720px] flex-1 p-4">
        <div className="arcade-panel rounded-xl p-6 text-center font-mono text-sm text-slate-400">
          No benchmark result JSON files found under benchmarks/results/*.json.
        </div>
      </main>
    )
  }

  return (
    <main className="mx-auto flex w-full max-w-[1720px] flex-1 flex-col gap-3.5 p-3 lg:p-4">
      {/* Arena Header + Provenance Badge + Static JSON Selector */}
      <section className="arcade-panel rounded-xl p-3.5">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <div className="flex flex-wrap items-center gap-2">
              <Trophy className="h-4 w-4 text-amber-300" />
              <h2 className="text-base font-bold tracking-wide text-white">
                Multi-VLA Head-to-Head Arena · 抗压榜单与破防回放馆
              </h2>
              <VerificationBadge verification={activeDataset.verification} />
              <span className="rounded border border-cyan-500/40 bg-cyan-500/15 px-2 py-0.5 font-mono text-[10px] font-semibold text-cyan-200">
                STATIC GITHUB PAGES READY
              </span>
            </div>
            <p className="mt-1 font-mono text-[11px] text-slate-400">
              {activeDataset.suite_name} · {activeDataset.hardware}
            </p>
            {activeDataset.notes && (
              <p className="mt-0.5 text-xs text-slate-400">{activeDataset.notes}</p>
            )}
          </div>

          {/* Dataset Picker & Local JSON Importer */}
          <div className="flex flex-wrap items-center gap-2">
            <div className="flex items-center gap-1.5 rounded-lg border border-slate-700 bg-slate-950/90 px-2.5 py-1.5 font-mono text-xs">
              <FileJson className="h-3.5 w-3.5 text-cyan-400" />
              <select
                aria-label="Select benchmark result JSON"
                value={activeDataset.fileName}
                onChange={(e) => {
                  setActiveFile(e.target.value)
                  setModelFilter('all')
                  setPrecisionFilter('all')
                  setSelectedIds([])
                  setContenderAId('')
                  setContenderBId('')
                }}
                className="bg-transparent text-xs text-slate-100 outline-none"
              >
                {datasets.map((ds) => (
                  <option
                    key={ds.fileName}
                    value={ds.fileName}
                    className="bg-slate-950 text-slate-100"
                  >
                    {datasetOptionPrefix(ds.verification)} {ds.sourcePath}
                  </option>
                ))}
              </select>
            </div>

            <input
              ref={fileInputRef}
              type="file"
              accept=".json,application/json"
              onChange={handleFileUpload}
              className="hidden"
            />
            <button
              type="button"
              onClick={() => fileInputRef.current?.click()}
              className="flex items-center gap-1.5 rounded-lg border border-cyan-400/50 bg-cyan-500/15 px-2.5 py-1.5 font-mono text-xs font-semibold text-cyan-200 transition hover:bg-cyan-500/25 active:scale-95"
            >
              <Upload className="h-3.5 w-3.5" />
              <span>LOAD JSON</span>
            </button>
          </div>
        </div>

        {uploadError && (
          <div className="mt-2 rounded-lg border border-rose-500/50 bg-rose-950/60 px-3 py-1.5 font-mono text-xs text-rose-200">
            Import error: {uploadError}
          </div>
        )}

        {/* KPI Strip + Filters */}
        <div className="mt-3 grid grid-cols-1 gap-2.5 border-t border-slate-800/90 pt-3 md:grid-cols-12">
          <div className="grid grid-cols-3 gap-2 md:col-span-6">
            <div className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2">
              <div className="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                <Trophy className="h-3 w-3 text-emerald-400" />
                TOP RESILIENT
              </div>
              <div className="mt-0.5 truncate font-mono text-xs font-bold text-emerald-300">
                {topScoreEntry
                  ? `${topScoreEntry.model} (${topScoreEntry.precision.toUpperCase()})`
                  : '-'}
              </div>
              <div className="font-mono text-[10px] text-slate-400">
                Score {topScoreEntry?.overall_score.toFixed(1) ?? '0'} / 100
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2">
              <div className="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                <Zap className="h-3 w-3 text-fuchsia-400" />
                FASTEST INFER
              </div>
              <div className="mt-0.5 truncate font-mono text-xs font-bold text-fuchsia-300">
                {fastestEntry
                  ? `${fastestEntry.model} (${fastestEntry.precision.toUpperCase()})`
                  : '-'}
              </div>
              <div className="font-mono text-[10px] text-slate-400">
                {fastestEntry
                  ? `${fastestEntry.latency.mean_ms.toFixed(1)}ms · ${fastestEntry.latency.fps.toFixed(1)}Hz`
                  : '-'}
              </div>
            </div>

            <div className="rounded-lg border border-slate-800 bg-slate-950/80 px-3 py-2">
              <div className="flex items-center gap-1 font-mono text-[10px] text-slate-400">
                <Database className="h-3 w-3 text-cyan-400" />
                CONFIGS LOADED
              </div>
              <div className="mt-0.5 font-mono text-sm font-bold text-white">
                {filteredEntries.length} / {activeDataset.entries.length}
              </div>
              <div className="font-mono text-[10px] text-slate-400">
                {activeDataset.replays.length} GIF comparisons
              </div>
            </div>
          </div>

          {/* Precision & Model Filters */}
          <div className="flex flex-wrap items-center justify-end gap-2 md:col-span-6">
            <div className="flex items-center gap-1 rounded-lg border border-slate-700/80 bg-slate-950/80 p-1 font-mono text-[10px]">
              <Filter className="mx-1 h-3 w-3 text-slate-400" />
              {PRECISION_FILTERS.map((pf) => (
                <button
                  key={pf.id}
                  type="button"
                  onClick={() => setPrecisionFilter(pf.id)}
                  className={`rounded px-2 py-1 transition ${
                    precisionFilter === pf.id
                      ? 'bg-cyan-500/25 font-bold text-cyan-200'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  {pf.label}
                </button>
              ))}
            </div>

            <div className="flex items-center gap-1.5 rounded-lg border border-slate-700/80 bg-slate-950/80 px-2.5 py-1.5 font-mono text-[11px]">
              <Cpu className="h-3.5 w-3.5 text-fuchsia-400" />
              <select
                aria-label="Filter by VLA model"
                value={modelFilter}
                onChange={(e) => setModelFilter(e.target.value)}
                className="bg-transparent text-xs text-slate-200 outline-none"
              >
                <option value="all" className="bg-slate-950">
                  All Models ({uniqueModels.length})
                </option>
                {uniqueModels.map((m) => (
                  <option key={m} value={m} className="bg-slate-950">
                    {m}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </section>

      {/* Head-to-Head 1v1 Comparison Card + Markdown Export */}
      <HeadToHeadCard
        dataset={activeDataset}
        entries={activeDataset.entries}
        contenderAId={effectiveContenderAId}
        contenderBId={effectiveContenderBId}
        onChangeContenderA={setContenderAId}
        onChangeContenderB={setContenderBId}
        onSwapContenders={handleSwapContenders}
      />

      {/* Chaos Survival Heatmap Matrix */}
      <ChaosSurvivalMatrix
        entries={filteredEntries}
        contenderAId={effectiveContenderAId}
        contenderBId={effectiveContenderBId}
        onPickContenderA={setContenderAId}
        onPickContenderB={setContenderBId}
      />

      {/* Charts: 5-Dim Radar + Latency Bars */}
      <div className="grid grid-cols-1 gap-3.5 xl:grid-cols-12">
        <div className="xl:col-span-5">
          <StressRadarChart entries={radarEntries} />
        </div>
        <div className="xl:col-span-7">
          <LatencyComparisonChart
            entries={filteredEntries}
            selectedIds={effectiveSelectedIds}
            onToggleSelect={toggleSelectEntry}
          />
        </div>
      </div>

      {/* Interactive Stress Resilience Leaderboard Table */}
      <section className="arcade-panel overflow-hidden rounded-xl">
        <div className="flex items-center justify-between border-b border-slate-800 px-3.5 py-2.5">
          <span className="font-mono text-xs font-bold tracking-wider text-slate-200">
            QUANTIZATION STRESS LEADERBOARD // 勾选行以叠加至五维雷达图
          </span>
          <span className="font-mono text-[10px] text-slate-400">
            SOURCE: {activeDataset.sourcePath}
          </span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full border-collapse text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-slate-800 bg-slate-950/80 text-[10px] uppercase tracking-wider text-slate-400">
                <th className="px-3 py-2">RADAR</th>
                <th className="px-3 py-2">MODEL</th>
                <th className="px-3 py-2">PREC</th>
                <th className="px-3 py-2">DENOISE</th>
                <th className="px-3 py-2 text-right">OVERALL</th>
                <th className="px-3 py-2 text-right">CLEAN</th>
                <th className="px-3 py-2 text-right">PHYS</th>
                <th className="px-3 py-2 text-right">VIS</th>
                <th className="px-3 py-2 text-right">SYS</th>
                <th className="px-3 py-2 text-right">QUANT</th>
                <th className="px-3 py-2 text-right">MEAN MS</th>
                <th className="px-3 py-2 text-right">VRAM GB</th>
                <th className="px-3 py-2 text-right">FPS</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/70">
              {filteredEntries.map((entry) => {
                const pinned = effectiveSelectedIds.includes(entry.id)
                return (
                  <tr
                    key={entry.id}
                    onClick={() => toggleSelectEntry(entry.id)}
                    className={`cursor-pointer transition ${
                      pinned ? 'bg-cyan-500/10' : 'hover:bg-slate-900/70'
                    }`}
                  >
                    <td className="px-3 py-2">
                      {pinned ? (
                        <CheckSquare className="h-4 w-4 text-cyan-300" />
                      ) : (
                        <Square className="h-4 w-4 text-slate-600" />
                      )}
                    </td>
                    <td className="px-3 py-2 font-semibold text-white">{entry.model}</td>
                    <td className="px-3 py-2 uppercase text-fuchsia-300">{entry.precision}</td>
                    <td className="px-3 py-2 text-slate-300">{entry.denoise_steps}</td>
                    <td className="px-3 py-2 text-right font-bold text-emerald-300">
                      {entry.overall_score.toFixed(1)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.radar_scores.clean}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.radar_scores.physical}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.radar_scores.visual}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.radar_scores.systems}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.radar_scores.quantization}
                    </td>
                    <td className="px-3 py-2 text-right font-semibold text-amber-300">
                      {entry.latency.mean_ms.toFixed(1)}
                    </td>
                    <td className="px-3 py-2 text-right text-slate-300">
                      {entry.vram_gb.toFixed(2)}
                    </td>
                    <td className="px-3 py-2 text-right text-cyan-300">
                      {entry.latency.fps.toFixed(1)}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Dual-Screen GIF Comparison Player */}
      <DualGifReplayPlayer replays={activeDataset.replays} />
    </main>
  )
}
