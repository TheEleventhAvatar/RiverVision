export type ImageQuality = { width: number; height: number; blur: number; brightness: number; accepted: boolean; message: string }

/** Browser-side intake checks. Blur score is a lightweight Laplacian proxy, not a trained quality model. */
export async function inspectImage(file: File): Promise<ImageQuality> {
  const bitmap = await createImageBitmap(file)
  const width = bitmap.width, height = bitmap.height
  const canvas = document.createElement('canvas'); canvas.width = 128; canvas.height = 128
  const ctx = canvas.getContext('2d', { willReadFrequently: true })!
  ctx.drawImage(bitmap, 0, 0, 128, 128); bitmap.close()
  const { data } = ctx.getImageData(0, 0, 128, 128)
  const gray = new Float32Array(128 * 128); let brightness = 0
  for (let i = 0; i < gray.length; i++) { const k = i * 4; gray[i] = .299 * data[k] + .587 * data[k + 1] + .114 * data[k + 2]; brightness += gray[i] }
  brightness /= gray.length
  let edge = 0
  for (let y = 1; y < 127; y++) for (let x = 1; x < 127; x++) { const i = y * 128 + x; edge += Math.abs(4 * gray[i] - gray[i - 1] - gray[i + 1] - gray[i - 128] - gray[i + 128]) }
  const blur = edge / (126 * 126)
  const lowResolution = width < 640 || height < 360
  const accepted = !(lowResolution || blur < 5.2 || brightness < 30)
  return { width, height, blur, brightness, accepted, message: lowResolution ? 'This image is below 640 × 360. Try a higher-resolution photo.' : blur < 5.2 ? 'This image appears very soft. Try a sharper photo.' : brightness < 30 ? 'This image is too dark to assess reliably. Try a brighter photo.' : 'Image quality looks suitable for a human-reviewed assessment.' }
}
