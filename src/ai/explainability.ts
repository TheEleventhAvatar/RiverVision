import type { Recommendations } from '../types'
export function explainImage(seed: number): Recommendations {
  return { channel: (requireChannel(seed)), water: requireWater(seed + 1), impervious: requireImpervious(seed + 2) }
}
import { classifyChannel as requireChannel } from './channelClassifier'
import { classifyWater as requireWater } from './waterClassifier'
import { detectImpervious as requireImpervious } from './imperviousDetector'
