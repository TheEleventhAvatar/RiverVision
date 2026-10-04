export const REVIEW_THRESHOLD = 0.65
export function requiresHumanReview(confidence: number, sufficientEvidence: boolean) { return confidence < REVIEW_THRESHOLD || !sufficientEvidence }
export function disagreement(ai: string, human: string) { return ai !== human }
