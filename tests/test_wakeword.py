import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from voice_assistant.wakeword import WakewordDetector  # noqa: E402


class TestWakewordDetector(unittest.TestCase):

    def test_below_threshold_never_triggers(self):
        det = WakewordDetector(threshold=0.5, required_consecutive=2)
        for i in range(10):
            d = det.feed(0.4, now=float(i))
            self.assertFalse(d.triggered)

    def test_requires_consecutive_hits(self):
        det = WakewordDetector(threshold=0.5, required_consecutive=2)
        d1 = det.feed(0.7, now=1.0)
        self.assertFalse(d1.triggered)
        self.assertEqual(d1.consecutive, 1)

        d2 = det.feed(0.7, now=1.1)
        self.assertTrue(d2.triggered)

    def test_below_threshold_resets_consecutive(self):
        det = WakewordDetector(threshold=0.5, required_consecutive=3)
        det.feed(0.7, now=1.0)
        det.feed(0.7, now=1.1)
        d_drop = det.feed(0.1, now=1.2)
        self.assertEqual(d_drop.consecutive, 0)
        # Need 3 fresh hits to trigger again
        det.feed(0.7, now=1.3)
        det.feed(0.7, now=1.4)
        d_final = det.feed(0.7, now=1.5)
        self.assertTrue(d_final.triggered)

    def test_low_average_blocks_trigger(self):
        # avg_factor 0.85 means avg must exceed 0.5 * 0.85 = 0.425.
        det = WakewordDetector(threshold=0.5, required_consecutive=2, window=5, avg_factor=0.85)
        # Pre-fill window with very low scores so the average stays low.
        for _ in range(4):
            det.feed(0.0, now=0.0)
        det.feed(0.7, now=1.0)
        d2 = det.feed(0.7, now=1.1)
        # avg over last 5 = (0+0+0+0.7+0.7)/5 = 0.28 < 0.425, so no trigger.
        self.assertFalse(d2.triggered)

    def test_cooldown_blocks_retrigger(self):
        det = WakewordDetector(threshold=0.5, required_consecutive=1, cooldown_seconds=2.0)
        d1 = det.feed(0.9, now=10.0)
        self.assertTrue(d1.triggered)

        d2 = det.feed(0.9, now=10.5)
        self.assertFalse(d2.triggered)
        self.assertTrue(d2.in_cooldown)

        d3 = det.feed(0.9, now=12.5)  # Just past cooldown
        self.assertTrue(d3.triggered)

    def test_reset_clears_state(self):
        det = WakewordDetector(threshold=0.5, required_consecutive=3)
        det.feed(0.7, now=1.0)
        det.feed(0.7, now=1.1)
        det.reset()
        # After reset, need 3 fresh hits again.
        det.feed(0.7, now=2.0)
        det.feed(0.7, now=2.1)
        d3 = det.feed(0.7, now=2.2)
        self.assertTrue(d3.triggered)

    def test_invalid_init_raises(self):
        with self.assertRaises(ValueError):
            WakewordDetector(threshold=0)
        with self.assertRaises(ValueError):
            WakewordDetector(threshold=0.5, window=0)


if __name__ == '__main__':
    unittest.main()
