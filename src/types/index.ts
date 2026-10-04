export type ChannelLabel = 'flat' | 'u' | 'v'
export type WaterLabel = 'clear' | 'muddy' | 'foam' | 'colored'
export type Suggestion<T> = { label: T; confidence: number; reason: string; alternative: string; evidenceRegions: string[]; sufficientEvidence: boolean }
export type ImperviousSuggestion = { prediction: 'yes' | 'no'; confidence: number; reason: string; alternative: string; detectedObjects: string[]; sufficientEvidence: boolean }
export type Recommendations = { channel: Suggestion<ChannelLabel>; water: Suggestion<WaterLabel>; impervious: ImperviousSuggestion }
export type AssessmentRecord = { id: string; fileName: string; timestamp: string; aiPrediction: Recommendations; humanDecision: Record<string, string>; finalDecision: Record<string, string>; disagreement: boolean }
