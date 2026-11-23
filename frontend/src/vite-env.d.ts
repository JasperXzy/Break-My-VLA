/// <reference types="vite/client" />

declare module 'virtual:benchmark-results' {
  export interface BundledBenchmarkArtifact {
    fileName: string
    sourcePath: string
    origin: 'repo' | 'public'
    raw: unknown
  }
  export const BUNDLED_BENCHMARKS: BundledBenchmarkArtifact[]
}
