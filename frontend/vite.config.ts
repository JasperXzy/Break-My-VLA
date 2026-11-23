import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import { benchmarkResultsPlugin } from './vite.benchmark-plugin'

const frontendRoot = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  base: './',
  plugins: [react(), tailwindcss(), benchmarkResultsPlugin(frontendRoot)],
  server: {
    host: true,
    port: 5173,
  },
})
