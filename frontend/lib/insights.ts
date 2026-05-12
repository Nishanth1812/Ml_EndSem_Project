export type RecommendationTone = 'positive' | 'warning' | 'danger' | 'neutral';

export type Recommendation = {
  label: string;
  tone: RecommendationTone;
  summary: string;
};

export function getRecommendation(predictedAlpha: number, beta: number): Recommendation {
  if (predictedAlpha > 0.002 && beta <= 1.2) {
    return {
      label: 'Strong buy',
      tone: 'positive',
      summary: 'Positive alpha with controlled market exposure.',
    };
  }

  if (predictedAlpha > 0 && beta <= 1.4) {
    return {
      label: 'Buy',
      tone: 'warning',
      summary: 'Positive edge with moderate beta risk.',
    };
  }

  if (predictedAlpha < -0.001) {
    return {
      label: 'Sell',
      tone: 'danger',
      summary: 'Negative alpha suggests weak forward edge.',
    };
  }

  return {
    label: 'Hold',
    tone: 'neutral',
    summary: 'Signal is neutral and should be treated conservatively.',
  };
}
