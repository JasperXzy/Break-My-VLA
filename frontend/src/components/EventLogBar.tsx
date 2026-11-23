import { useState } from 'react'
import { ChevronDown, ChevronUp, Terminal, Trash2 } from 'lucide-react'
import type { EventLogEntry } from '../types/protocol'

interface EventLogBarProps {
  logs: EventLogEntry[]
  onClear: () => void
}

const BADGE_STYLE: Record<EventLogEntry['category'], string> = {
  chaos: 'border-rose-400/50 bg-rose-500/20 text-rose-200',
  config: 'border-fuchsia-400/50 bg-fuchsia-500/20 text-fuchsia-200',
  control: 'border-cyan-400/50 bg-cyan-500/20 text-cyan-200',
  status: 'border-emerald-400/50 bg-emerald-500/20 text-emerald-200',
  system: 'border-amber-400/50 bg-amber-500/20 text-amber-200',
  error: 'border-red-500/60 bg-red-500/25 text-red-200',
}

export function EventLogBar({ logs, onClear }: EventLogBarProps) {
  const [expanded, setExpanded] = useState(false)

  return (
    <footer className="arcade-panel border-t border-slate-700/80 px-3 py-2 lg:px-4">
      <div className="flex items-center justify-between gap-2 pb-1.5">
        <div className="flex items-center gap-2 font-mono text-[11px] font-semibold tracking-wider text-cyan-300">
          <Terminal className="h-3.5 w-3.5" />
          <span>EVENT TELEMETRY STREAM</span>
          <span className="rounded bg-slate-800 px-1.5 py-0.5 text-[10px] text-slate-400">
            {logs.length} EVENTS
          </span>
        </div>

        <div className="flex items-center gap-1.5 font-mono text-[10px]">
          <button
            type="button"
            onClick={onClear}
            className="flex items-center gap-1 rounded border border-slate-700/80 bg-slate-900 px-2 py-0.5 text-slate-400 transition hover:border-slate-500 hover:text-slate-200"
          >
            <Trash2 className="h-3 w-3" />
            CLEAR
          </button>
          <button
            type="button"
            onClick={() => setExpanded((e) => !e)}
            className="flex items-center gap-1 rounded border border-slate-700/80 bg-slate-900 px-2 py-0.5 text-slate-300 transition hover:border-cyan-400/60 hover:text-cyan-200"
          >
            {expanded ? <ChevronDown className="h-3 w-3" /> : <ChevronUp className="h-3 w-3" />}
            {expanded ? 'COMPACT' : 'EXPAND'}
          </button>
        </div>
      </div>

      <div
        className={`overflow-y-auto rounded-lg border border-slate-800/90 bg-slate-950/90 px-2.5 py-1.5 font-mono text-[11px] transition-[max-height] duration-200 ${
          expanded ? 'max-h-48' : 'max-h-20'
        }`}
      >
        {logs.length === 0 ? (
          <div className="py-1 text-slate-500">
            Awaiting chaos injections or robot status transitions...
          </div>
        ) : (
          <ul className="space-y-1">
            {logs.map((entry) => (
              <li
                key={entry.id}
                className="flex flex-wrap items-center gap-x-2 gap-y-0.5 border-b border-slate-900/80 pb-1 last:border-b-0 last:pb-0"
              >
                <span className="text-[10px] text-slate-500">[{entry.timestamp}]</span>
                <span
                  className={`rounded border px-1.5 py-0 text-[9px] font-bold uppercase ${BADGE_STYLE[entry.category]}`}
                >
                  {entry.category}
                </span>
                <span className="font-semibold text-slate-100">{entry.title}</span>
                {entry.detail && (
                  <span className="truncate text-slate-400">{entry.detail}</span>
                )}
              </li>
            ))}
          </ul>
        )}
      </div>
    </footer>
  )
}
