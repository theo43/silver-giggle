import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path
from translation.dataset import causal_mask


BASE_PATH = Path(__file__).resolve().parent


class TestCausalMask(unittest.TestCase):
    def test_create_causal_mask_size_3(self):
        expected_mask = [[[1, 0, 0], [1, 1, 0], [1, 1, 1]]]
        mask = causal_mask(3)

        self.assertEqual(mask.tolist(), expected_mask)
