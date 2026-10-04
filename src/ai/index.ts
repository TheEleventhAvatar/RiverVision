let readyPromise: Promise<boolean> | undefined
export function initializeVision(): Promise<boolean> {
  if (!readyPromise) readyPromise = (async () => {
    try {
      // Keep TensorFlow out of the initial UI bundle while preserving a real local WebGL runtime.
      await import('@tensorflow/tfjs-backend-webgl')
      const tf = await import('@tensorflow/tfjs')
      await tf.setBackend('webgl'); await tf.ready(); return tf.getBackend() === 'webgl'
    }
    catch { return false }
  })()
  return readyPromise
}
export { explainImage } from './explainability'
export { inspectImage } from './imageQuality'
