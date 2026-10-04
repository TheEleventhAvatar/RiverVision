import type { Suggestion, ChannelLabel } from '../types'

// TODO: Replace demo heuristic with a trained, validated stream morphology model. Never use this score as a finding.
export function classifyChannel(seed = 1): Suggestion<ChannelLabel> {
  const n = (seed % 3); const label: ChannelLabel = (['u', 'v', 'flat'] as const)[n]
  return { label, confidence: [.76, .68, .61][n], reason: label === 'v' ? 'Bank edges appear to converge toward the visible channel.' : label === 'u' ? 'The visible banks appear rounded near the waterline.' : 'The visible channel appears relatively broad and low-sided.', alternative: 'Camera angle, vegetation, or an obscured bank can change the apparent channel shape.', evidenceRegions: ['channel banks'], sufficientEvidence: true }
}
