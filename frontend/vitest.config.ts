import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vitest/config'
import react from '@vitejs/plugin-react'
import { benchmarkResultsPlugin } from './vite.benchmark-plugin'

const frontendRoot = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react(), benchmarkResultsPlugin(frontendRoot)],
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
  },
})
