/**
 * Zero-dependency GIF download + lightweight GIF89a animated encoder
 * for Local Demo Mode replays.
 */

export function triggerBlobDownload(blob: Blob, filename: string): void {
  const url = URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  document.body.appendChild(anchor)
  anchor.click()
  document.body.removeChild(anchor)
  setTimeout(() => URL.revokeObjectURL(url), 1500)
}

export function downloadBase64Gif(b64: string, filename?: string): void {
  const cleaned = b64.includes(',') ? b64.split(',')[1] : b64
  const binary = atob(cleaned)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  const blob = new Blob([bytes], { type: 'image/gif' })
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)
  triggerBlobDownload(blob, filename ?? `break-my-vla-replay-${stamp}.gif`)
}

/** Build a fixed 256-color global palette: 6x6x6 RGB cube (216) + 40 grayscale ramp. */
function buildGlobalPalette(): Uint8Array {
  const palette = new Uint8Array(256 * 3)
  let idx = 0
  for (let r = 0; r < 6; r++) {
    for (let g = 0; g < 6; g++) {
      for (let b = 0; b < 6; b++) {
        palette[idx++] = Math.round((r * 255) / 5)
        palette[idx++] = Math.round((g * 255) / 5)
        palette[idx++] = Math.round((b * 255) / 5)
      }
    }
  }
  for (let i = 0; i < 40; i++) {
    const v = Math.round((i * 255) / 39)
    palette[idx++] = v
    palette[idx++] = v
    palette[idx++] = v
  }
  return palette
}

function quantizeFrame(imageData: ImageData): Uint8Array {
  const { data, width, height } = imageData
  const indices = new Uint8Array(width * height)
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const r = data[i]
    const g = data[i + 1]
    const b = data[i + 2]
    const ri = Math.round((r * 5) / 255)
    const gi = Math.round((g * 5) / 255)
    const bi = Math.round((b * 5) / 255)
    indices[p] = ri * 36 + gi * 6 + bi
  }
  return indices
}

function lzwEncode(indices: Uint8Array, minCodeSize = 8): Uint8Array {
  const clearCode = 1 << minCodeSize
  const eoiCode = clearCode + 1
  let codeSize = minCodeSize + 1
  let nextCode = eoiCode + 1

  const out: number[] = []
  let bitBuf = 0
  let bitCount = 0

  const emit = (code: number) => {
    bitBuf |= code << bitCount
    bitCount += codeSize
    while (bitCount >= 8) {
      out.push(bitBuf & 0xff)
      bitBuf >>= 8
      bitCount -= 8
    }
  }

  let table = new Map<number, number>()
  const resetTable = () => {
    table = new Map<number, number>()
    codeSize = minCodeSize + 1
    nextCode = eoiCode + 1
  }

  emit(clearCode)
  let prefix = indices[0] ?? 0

  for (let i = 1; i < indices.length; i++) {
    const k = indices[i]
    const key = (prefix << 8) | k
    const found = table.get(key)
    if (found !== undefined) {
      prefix = found
    } else {
      emit(prefix)
      if (nextCode < 4096) {
        table.set(key, nextCode++)
        if (nextCode > (1 << codeSize) && codeSize < 12) {
          codeSize++
        }
      } else {
        emit(clearCode)
        resetTable()
      }
      prefix = k
    }
  }

  emit(prefix)
  emit(eoiCode)
  if (bitCount > 0) {
    out.push(bitBuf & 0xff)
  }

  return new Uint8Array(out)
}

function writeU16LE(buf: number[], value: number): void {
  buf.push(value & 0xff, (value >> 8) & 0xff)
}

export function encodeFramesToGifBlob(frames: ImageData[], delayCs = 8): Blob {
  if (frames.length === 0) {
    throw new Error('No frames available to export.')
  }
  const width = frames[0].width
  const height = frames[0].height
  const palette = buildGlobalPalette()
  const bytes: number[] = []

  // Header: GIF89a
  bytes.push(0x47, 0x49, 0x46, 0x38, 0x39, 0x61)

  // Logical Screen Descriptor
  writeU16LE(bytes, width)
  writeU16LE(bytes, height)
  // Global color table flag (1), color res (7), sort (0), size (7 -> 256 colors)
  bytes.push(0xf7, 0x00, 0x00)

  // Global Color Table
  for (let i = 0; i < palette.length; i++) {
    bytes.push(palette[i])
  }

  // NETSCAPE2.0 looping application extension
  bytes.push(
    0x21, 0xff, 0x0b,
    0x4e, 0x45, 0x54, 0x53, 0x43, 0x41, 0x50, 0x45, 0x32, 0x2e, 0x30,
    0x03, 0x01, 0x00, 0x00, 0x00,
  )

  for (const frame of frames) {
    // Graphics Control Extension
    bytes.push(0x21, 0xf9, 0x04, 0x04)
    writeU16LE(bytes, delayCs)
    bytes.push(0x00, 0x00)

    // Image Descriptor
    bytes.push(0x2c)
    writeU16LE(bytes, 0)
    writeU16LE(bytes, 0)
    writeU16LE(bytes, width)
    writeU16LE(bytes, height)
    bytes.push(0x00)

    // LZW Image Data
    const minCodeSize = 8
    bytes.push(minCodeSize)
    const indexed = quantizeFrame(frame)
    const compressed = lzwEncode(indexed, minCodeSize)
    let offset = 0
    while (offset < compressed.length) {
      const chunkLen = Math.min(255, compressed.length - offset)
      bytes.push(chunkLen)
      for (let i = 0; i < chunkLen; i++) {
        bytes.push(compressed[offset + i])
      }
      offset += chunkLen
    }
    bytes.push(0x00)
  }

  // Trailer
  bytes.push(0x3b)
  return new Blob([new Uint8Array(bytes)], { type: 'image/gif' })
}
