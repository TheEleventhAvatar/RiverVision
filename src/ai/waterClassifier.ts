import type { Suggestion, WaterLabel } from '../types'

// TODO: Replace with a water-quality appearance classifier trained on diverse local conditions.
export function classifyWater(seed = 1): Suggestion<WaterLabel> {
  const label: WaterLabel = (['clear', 'muddy', 'colored', 'foam'] as const)[seed % 4]
  return { label, confidence: [.71, .79, .58, .66][seed % 4], reason: label === 'muddy' ? 'The visible water region has a low-clarity, sediment-like appearance.' : `The visible water surface has appearance cues consistent with ${label === 'clear' ? 'higher clarity' : label}.`, alternative: 'Reflections, lighting, depth, and camera white balance can change water appearance; this is not a water-quality test.', evidenceRegions: ['visible water surface'], sufficientEvidence: true }
}
