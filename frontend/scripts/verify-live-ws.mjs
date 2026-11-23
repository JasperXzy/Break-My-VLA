/**
 * End-to-end Live stack verifier for Break-My-VLA.
 * Exercises mock_server (8001), chaos_sim (8000), and Vite dev server (5173)
 * using the exact frontend WebSocket wire protocol.
 */

import WebSocket from 'ws'

const POLICY_URL = process.env.POLICY_URL || 'http://127.0.0.1:8001'
const SIM_HTTP_URL = process.env.SIM_HTTP_URL || 'http://127.0.0.1:8000'
const SIM_WS_URL = process.env.SIM_WS_URL || 'ws://127.0.0.1:8000/ws/session'
const VITE_URL = process.env.VITE_URL || 'http://127.0.0.1:5173'

function assert(condition, message) {
  if (!condition) {
    throw new Error(`ASSERTION FAILED: ${message}`)
  }
}

class WsHarness {
  constructor(url) {
    this.url = url
    this.messages = []
    this.errors = []
    this.waiters = []
  }

  async connect() {
    await new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.url)
      this.ws.once('open', resolve)
      this.ws.once('error', reject)
      this.ws.on('message', (data) => {
        const msg = JSON.parse(data.toString())
        this.messages.push(msg)
        if (msg.type === 'error') {
          this.errors.push(msg)
        }
        for (let i = this.waiters.length - 1; i >= 0; i--) {
          const waiter = this.waiters[i]
          if (waiter.predicate(msg)) {
            clearTimeout(waiter.timer)
            this.waiters.splice(i, 1)
            waiter.resolve(msg)
          }
        }
      })
    })
  }

  send(payload) {
    this.ws.send(JSON.stringify(payload))
  }

  waitFor(predicate, label, timeoutMs = 6000) {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        reject(new Error(`Timed out waiting for: ${label}`))
      }, timeoutMs)
      this.waiters.push({ predicate, resolve, timer })
    })
  }

  async close() {
    if (!this.ws) return
    await new Promise((resolve) => {
      if (this.ws.readyState === WebSocket.CLOSED) {
        resolve()
        return
      }
      this.ws.once('close', resolve)
      this.ws.close()
    })
  }
}

async function run() {
  console.log('1) Checking mock_server /health and Vite dev server...')
  const healthRes = await fetch(`${POLICY_URL}/health`)
  assert(healthRes.ok, `mock_server /health failed: ${healthRes.status}`)
  const healthJson = await healthRes.json()
  assert(healthJson.model === 'mock', `unexpected policy model: ${healthJson.model}`)
  console.log('   OK mock_server:', JSON.stringify(healthJson))

  const indexRes = await fetch(`${VITE_URL}/`)
  assert(indexRes.ok, `Vite / returned ${indexRes.status}`)
  const indexHtml = await indexRes.text()
  assert(indexHtml.includes('BREAK-MY-VLA'), 'Vite index.html missing BREAK-MY-VLA title')

  const manifestRes = await fetch(`${VITE_URL}/benchmarks/results/index.json`)
  assert(manifestRes.ok, `Manifest returned ${manifestRes.status}`)
  const manifest = await manifestRes.json()
  assert(
    Array.isArray(manifest.files) && manifest.files.includes('auto_chaos_m4_showcase.json'),
    `Unexpected benchmark manifest: ${JSON.stringify(manifest)}`,
  )

  const benchRes = await fetch(`${VITE_URL}/benchmarks/results/auto_chaos_m4_showcase.json`)
  assert(benchRes.ok, `Benchmark JSON returned ${benchRes.status}`)
  const benchJson = await benchRes.json()
  assert(
    Array.isArray(benchJson.entries) && benchJson.entries.length >= 8,
    'Benchmark JSON missing expected entries',
  )

  const gifAssetRes = await fetch(`${VITE_URL}/replays/bf16_clean_pass.gif`)
  assert(gifAssetRes.ok, `Replay GIF returned ${gifAssetRes.status}`)
  const gifAssetBuf = Buffer.from(await gifAssetRes.arrayBuffer())
  assert(
    gifAssetBuf.subarray(0, 3).toString('ascii') === 'GIF',
    'Replay GIF asset does not have GIF magic header',
  )
  console.log(
    `   OK Vite server (${VITE_URL}) serving HTML, ${benchJson.entries.length} benchmark entries, and valid GIF (${gifAssetBuf.length} bytes)`,
  )

  console.log('2) Connecting to Live chaos_sim WebSocket...')
  const client = new WsHarness(SIM_WS_URL)
  await client.connect()

  const firstLive = await client.waitFor(
    (m) => m.type === 'frame' && m.step >= 2 && m.action_chunk.length > 0,
    'initial streaming frames',
  )
  assert(firstLive.rgb_main_b64.length > 100, 'rgb_main_b64 too short')
  assert(firstLive.rgb_wrist_b64.length > 100, 'rgb_wrist_b64 too short')
  console.log(
    `   OK live stream active (step=${firstLive.step}, fps=${firstLive.fps.toFixed(1)}, infer_ms=${firstLive.infer_ms.toFixed(1)})`,
  )

  console.log('3) Verifying all 8 chaos skills...')
  const skills = [
    { skill: 'object_nudge', params: { dx: 0.18, dy: -0.12, dz: 0.0 } },
    { skill: 'arm_kick', params: { dx: -0.22, dy: 0.1, dz: 0.12 } },
    { skill: 'friction_glitch', params: { duration_steps: 45, multiplier: 0.1 } },
    { skill: 'lighting_shift', params: { duration_steps: 40, brightness: 0.45, saturation: 0.65 } },
    { skill: 'camera_shake', params: { duration_steps: 36, amplitude_px: 16 } },
    {
      skill: 'occlusion_patch',
      params: { duration_steps: 45, x: 0.36, y: 0.3, width: 0.29, height: 0.25 },
    },
    { skill: 'latency_inject', params: { duration_steps: 40, latency_ms: 90 } },
    { skill: 'wrist_blackout', params: { duration_steps: 40 } },
  ]

  for (const item of skills) {
    client.send({ type: 'chaos', skill: item.skill, params: item.params })
    const observed = await client.waitFor(
      (m) => m.type === 'frame' && Array.isArray(m.active_chaos) && m.active_chaos.includes(item.skill),
      `active_chaos includes ${item.skill}`,
    )
    console.log(`   OK chaos skill '${item.skill}' observed at step=${observed.step}`)
  }

  console.log('4) Verifying precision hot-swap (bf16/fp8/int8/nvfp4) & denoise steps slider...')
  for (const precision of ['fp8', 'int8', 'nvfp4', 'bf16']) {
    client.send({ type: 'config', precision })
    const f = await client.waitFor(
      (m) => m.type === 'frame' && m.precision === precision,
      `precision=${precision}`,
    )
    console.log(`   OK precision -> ${f.precision} (infer_ms=${f.infer_ms.toFixed(1)})`)
  }

  for (const denoise_steps of [3, 1, 8]) {
    client.send({ type: 'config', denoise_steps })
    const f = await client.waitFor(
      (m) => m.type === 'frame' && m.denoise_steps === denoise_steps,
      `denoise_steps=${denoise_steps}`,
    )
    console.log(`   OK denoise_steps -> ${f.denoise_steps}`)
  }

  client.send({ type: 'config', latency_ms: 80, prompt: 'Lift the orange block cleanly.' })
  await client.waitFor((m) => m.type === 'frame' && m.infer_ms >= 80, 'config latency_ms >= 80')
  client.send({ type: 'config', latency_ms: 0 })
  console.log('   OK config latency_ms & prompt')

  console.log('5) Verifying Pause / Resume / Reset / Export GIF...')
  client.send({ type: 'control', action: 'pause' })
  const paused = await client.waitFor((m) => m.type === 'frame' && m.status === 'paused', 'paused frame')
  console.log(`   OK pause at step=${paused.step}`)

  client.send({ type: 'control', action: 'resume' })
  const resumed = await client.waitFor(
    (m) => m.type === 'frame' && m.status !== 'paused' && m.step > paused.step,
    'resumed stepping',
  )
  console.log(`   OK resume -> step=${resumed.step}`)

  client.send({ type: 'control', action: 'export_gif' })
  const gifMsg = await client.waitFor((m) => m.type === 'gif', 'WS gif message', 10000)
  const gifBytes = Buffer.from(gifMsg.gif_b64, 'base64')
  const gifHeader = gifBytes.subarray(0, 6).toString('ascii')
  assert(gifHeader === 'GIF89a' || gifHeader === 'GIF87a', `Invalid GIF header: ${gifHeader}`)
  assert(gifBytes.length > 1000, `GIF too small: ${gifBytes.length} bytes`)
  console.log(`   OK WS export_gif (${gifHeader}, ${gifBytes.length} bytes)`)

  const httpGif = await fetch(`${SIM_HTTP_URL}/api/export_gif`, { method: 'POST' })
  assert(httpGif.ok, `HTTP /api/export_gif failed: ${httpGif.status}`)
  const httpBuf = Buffer.from(await httpGif.arrayBuffer())
  assert(httpBuf.subarray(0, 3).toString('ascii') === 'GIF', 'HTTP GIF invalid header')
  console.log(`   OK HTTP POST /api/export_gif (${httpBuf.length} bytes)`)

  client.send({ type: 'control', action: 'reset' })
  const resetFrame = await client.waitFor((m) => m.type === 'frame' && m.step === 0, 'reset step=0')
  console.log(`   OK reset -> step=${resetFrame.step}, active_chaos=${JSON.stringify(resetFrame.active_chaos)}`)

  assert(client.errors.length === 0, `Received unexpected server errors: ${JSON.stringify(client.errors)}`)

  console.log('6) Verifying disconnect & reconnect recovery...')
  await client.close()
  const reconnected = new WsHarness(SIM_WS_URL)
  await reconnected.connect()
  const freshFrame = await reconnected.waitFor(
    (m) => m.type === 'frame' && m.step >= 2 && m.status === 'running',
    'reconnected streaming frames',
  )
  reconnected.send({ type: 'chaos', skill: 'wrist_blackout', params: { duration_steps: 20 } })
  await reconnected.waitFor(
    (m) => m.type === 'frame' && m.active_chaos.includes('wrist_blackout'),
    'chaos after reconnect',
  )
  assert(reconnected.errors.length === 0, `Errors after reconnect: ${JSON.stringify(reconnected.errors)}`)
  await reconnected.close()
  console.log(`   OK disconnect & reconnect verified (fresh step=${freshFrame.step}, 0 errors)`)

  console.log('\nALL LIVE E2E CHECKS PASSED WITH ZERO ERRORS!')
}

run().catch((err) => {
  console.error(err)
  process.exit(1)
})
