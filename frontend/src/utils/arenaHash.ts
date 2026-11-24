import type { AppView } from '../types/protocol'

export interface ArenaDeepLinkState {
  view: AppView
  hasExplicitView: boolean
  dataset?: string
  a?: string
  b?: string
}

/**
 * Parses URL hashes such as:
 * - `#arcade`
 * - `#gallery`
 * - `#gallery?dataset=auto_chaos_m4_showcase.json&a=groot-bf16&b=groot-nvfp4-trt`
 */
export function parseArenaHash(rawHash: string): ArenaDeepLinkState {
  const trimmed = rawHash.replace(/^#/, '').trim()
  if (!trimmed) {
    return { view: 'arcade', hasExplicitView: false }
  }

  const qIdx = trimmed.indexOf('?')
  const route = (qIdx >= 0 ? trimmed.slice(0, qIdx) : trimmed).toLowerCase()
  const queryStr = qIdx >= 0 ? trimmed.slice(qIdx + 1) : ''

  if (route === 'gallery') {
    const params = new URLSearchParams(queryStr)
    const dataset = params.get('dataset')?.trim() || undefined
    const a = params.get('a')?.trim() || undefined
    const b = params.get('b')?.trim() || undefined
    return {
      view: 'gallery',
      hasExplicitView: true,
      dataset,
      a,
      b,
    }
  }

  if (route === 'arcade') {
    return { view: 'arcade', hasExplicitView: true }
  }

  return { view: 'arcade', hasExplicitView: false }
}

export function buildArenaShareHash(params: {
  dataset: string
  a: string
  b: string
}): string {
  const qs = new URLSearchParams()
  if (params.dataset) qs.set('dataset', params.dataset)
  if (params.a) qs.set('a', params.a)
  if (params.b) qs.set('b', params.b)
  const query = qs.toString()
  return query ? `#gallery?${query}` : '#gallery'
}

export function buildArenaShareUrl(params: {
  dataset: string
  a: string
  b: string
}): string {
  const hash = buildArenaShareHash(params)
  if (typeof window === 'undefined') return hash
  const { origin, pathname, search } = window.location
  const base = origin && origin !== 'null' ? `${origin}${pathname}${search}` : `${pathname}${search}`
  return `${base}${hash}`
}
