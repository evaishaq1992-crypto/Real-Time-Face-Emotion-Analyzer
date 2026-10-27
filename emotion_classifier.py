import numpy as np
from dataclasses import dataclass, field
from utils import MovingAverage

EMOTIONS = {
    'happy': '😄', 'sad': '😢', 'angry': '😡',
    'surprise': '😲', 'neutral': '😐'
}

@dataclass
class EmotionResult:
    name: str = 'neutral'
    emoji: str = '😐'
    confidence: float = 0.0
    scores: dict = field(default_factory=dict)

class EmotionClassifier:
    def __init__(self, alpha=0.25):
        self.ema_ear = MovingAverage(alpha)
        self.ema_mar = MovingAverage(alpha)
        self.ema_smile = MovingAverage(alpha)
        self.ema_brow = MovingAverage(alpha)
        self.ema_fhd = MovingAverage(alpha)
        self.cal_buf = []
        self.cal_n = 0
        self.baseline_smile = None

    def classify(self, feat):
        s_ear = self.ema_ear.update(feat.ear)
        s_mar = self.ema_mar.update(feat.mar)
        s_smile = self.ema_smile.update(feat.smile)
        s_brow = self.ema_brow.update(feat.brow_dist)
        s_fhd = self.ema_fhd.update(feat.fhd_var)

        if self.cal_n < 30:
            self.cal_buf.append(s_smile)
            self.cal_n += 1
            return EmotionResult(name='neutral', emoji='😐', confidence=1.0)
        
        if self.baseline_smile is None:
            self.baseline_smile = np.mean(self.cal_buf)

        # Scoring (0.0 to 1.0)
        scores = {}
        delta_smile = s_smile - self.baseline_smile
        scores['happy'] = float(np.clip(delta_smile / 0.12, 0, 1))
        
        scores['surprise'] = float(np.clip(
            0.5 * ((s_ear - 0.28) / 0.06) + 0.5 * ((s_mar - 0.4) / 0.2), 0, 1))
        
        brow_furrow = float(np.clip((0.18 - s_brow) / 0.04, 0, 1))
        fhd_score = float(np.clip((s_fhd - 800) / 1000, 0, 1))
        scores['angry'] = float(np.clip(0.6 * brow_furrow + 0.4 * fhd_score, 0, 1))
        
        low_smile = float(np.clip((self.baseline_smile - 0.03 - s_smile) / 0.06, 0, 1))
        narrow_eye = float(np.clip((0.22 - s_ear) / 0.04, 0, 1))
        scores['sad'] = float(np.clip(0.7 * low_smile + 0.3 * narrow_eye, 0, 1))
        
        scores['neutral'] = float(np.clip(1.0 - max(scores['happy'], scores['surprise'], scores['angry'], scores['sad']), 0, 1))

        best_emo = 'neutral'
        best_conf = scores['neutral']
        for emo, sc in scores.items():
            if emo != 'neutral' and sc > best_conf and sc > 0.25:
                best_emo, best_conf = emo, sc

        return EmotionResult(name=best_emo, emoji=EMOTIONS[best_emo], confidence=best_conf, scores=scores)

    def reset(self):
        self.ema_ear.value = None
        self.ema_mar.value = None
        self.ema_smile.value = None
        self.ema_brow.value = None
        self.ema_fhd.value = None
        self.cal_buf = []
        self.cal_n = 0
        self.baseline_smile = None