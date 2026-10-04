import type { ImperviousSuggestion } from '../types'

// TODO: Replace with a validated segmentation model. Detection is a demo suggestion only.
export function detectImpervious(seed = 1): ImperviousSuggestion {
  const yes = seed % 2 === 0
  return { prediction: yes ? 'yes' : 'no', confidence: yes ? .72 : .64, reason: yes ? 'A hard-edged surface may be visible along the left margin.' : 'No clear continuous hard surface is visible along the left margin.', alternative: 'Partial framing, shadows, or vegetation may obscure roads and built surfaces.', detectedObjects: yes ? ['road'] : [], sufficientEvidence: true }
}
