"""
emotions - Real-time facial emotion detection and filter auto-suggestion.
"""

from emotions.emotion_detector import (
    BaseEmotionClassifier,
    RuleBasedEmotionClassifier,
    EmotionSmoother,
    EmotionDetector,
)

__all__ = [
    "BaseEmotionClassifier",
    "RuleBasedEmotionClassifier",
    "EmotionSmoother",
    "EmotionDetector",
]
