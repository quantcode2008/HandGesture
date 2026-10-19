"""
test_emotion_detector.py - Unit tests for facial emotion detection rule scorer,
interchangeable classifier interface, and temporal smoother.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from emotions.emotion_detector import (
    BaseEmotionClassifier,
    EmotionSmoother,
    RuleBasedEmotionClassifier,
)


class TestEmotionRuleScorer(unittest.TestCase):
    def setUp(self):
        self.classifier = RuleBasedEmotionClassifier()

    def test_happy_detection(self):
        """Strong smile blendshapes should classify as Happy with high confidence."""
        blendshapes = {
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.85,
            "cheekSquintLeft": 0.30,
            "cheekSquintRight": 0.30,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Happy")
        self.assertGreater(conf, 0.70)

    def test_sad_detection(self):
        """Frown and inner brow raise should classify as Sad."""
        blendshapes = {
            "mouthFrownLeft": 0.75,
            "mouthFrownRight": 0.75,
            "browInnerUp": 0.60,
            "mouthShrugLower": 0.30,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Sad")
        self.assertGreater(conf, 0.50)

    def test_angry_detection(self):
        """Furrowed brows down and mouth press should classify as Angry."""
        blendshapes = {
            "browDownLeft": 0.80,
            "browDownRight": 0.80,
            "mouthPressLeft": 0.40,
            "mouthPressRight": 0.40,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Angry")
        self.assertGreater(conf, 0.50)

    def test_surprised_detection(self):
        """Open jaw, raised brows, and wide eyes should classify as Surprised."""
        blendshapes = {
            "jawOpen": 0.85,
            "eyeWideLeft": 0.70,
            "eyeWideRight": 0.70,
            "browInnerUp": 0.60,
            "browOuterUpLeft": 0.50,
            "browOuterUpRight": 0.50,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Surprised")
        self.assertGreater(conf, 0.65)

    def test_fear_detection(self):
        """Wide eyes, inner brow raise, and mouth stretch should classify as Fear."""
        blendshapes = {
            "eyeWideLeft": 0.80,
            "eyeWideRight": 0.80,
            "browInnerUp": 0.70,
            "mouthStretchLeft": 0.60,
            "mouthStretchRight": 0.60,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Fear")
        self.assertGreater(conf, 0.50)

    def test_disgust_detection(self):
        """Nose sneer and upper lip raise should classify as Disgust."""
        blendshapes = {
            "noseSneerLeft": 0.85,
            "noseSneerRight": 0.85,
            "mouthUpperUpLeft": 0.60,
            "mouthUpperUpRight": 0.60,
            "browDownLeft": 0.30,
            "browDownRight": 0.30,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Disgust")
        self.assertGreater(conf, 0.50)

    def test_neutral_detection(self):
        """Subtle/resting facial features below threshold should classify as Neutral."""
        blendshapes = {
            "mouthSmileLeft": 0.05,
            "mouthSmileRight": 0.05,
            "browDownLeft": 0.04,
            "browDownRight": 0.04,
            "jawOpen": 0.02,
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Neutral")
        self.assertGreaterEqual(conf, 0.50)

    def test_negative_weight_suppression(self):
        """Smile should suppress Sad even if a small frown or brow-up is present."""
        blendshapes = {
            "mouthSmileLeft": 0.90,
            "mouthSmileRight": 0.90,
            "browInnerUp": 0.30,  # Slight brow raise, but smiling
        }
        label, conf = self.classifier.predict(blendshapes=blendshapes)
        self.assertEqual(label, "Happy")

    def test_empty_blendshapes_safe(self):
        """Empty blendshapes dict should safely return Neutral without exceptions."""
        label, conf = self.classifier.predict(blendshapes={})
        self.assertEqual(label, "Neutral")
        self.assertGreaterEqual(conf, 0.50)


class TestCustomClassifierInterface(unittest.TestCase):
    def test_custom_classifier_subclass(self):
        """Custom classifier implementing BaseEmotionClassifier can be swapped in."""
        class MockMLClassifier(BaseEmotionClassifier):
            def predict(self, frame=None, landmarks=None, blendshapes=None):
                return "MockJoy", 0.99

        custom = MockMLClassifier()
        label, conf = custom.predict(frame=None, landmarks=None, blendshapes={})
        self.assertEqual(label, "MockJoy")
        self.assertEqual(conf, 0.99)


class TestEmotionSmoother(unittest.TestCase):
    def setUp(self):
        self.smoother = EmotionSmoother(window_size=5)

    def test_smoother_stops_single_frame_flicker(self):
        """A single outlier frame in a steady sequence must not flip the smoothed result."""
        # 4 frames of Happy
        for _ in range(4):
            self.smoother.add("Happy", 0.90)

        # 1 outlier frame of Sad (e.g. tracking glitch)
        smoothed_label, smoothed_conf = self.smoother.add("Sad", 0.80)

        # Majority vote should keep it as Happy
        self.assertEqual(smoothed_label, "Happy")
        self.assertGreater(smoothed_conf, 0.70)

    def test_smoother_transitions_on_sustained_change(self):
        """Sustained change over multiple frames transitions to the new emotion."""
        for _ in range(3):
            self.smoother.add("Happy", 0.80)

        # Sustained shift to Surprised
        for _ in range(4):
            lbl, _ = self.smoother.add("Surprised", 0.95)

        self.assertEqual(lbl, "Surprised")

    def test_smoother_reset(self):
        """reset() should clear the window."""
        self.smoother.add("Happy", 0.90)
        self.smoother.reset()
        lbl, _ = self.smoother.get_smoothed()
        self.assertEqual(lbl, "Neutral")


if __name__ == "__main__":
    unittest.main()
