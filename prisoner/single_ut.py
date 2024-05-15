import unittest

from prisoner import single_round

COOPERATE = 1
DEFECT = 0


class TestSingleRound(unittest.TestCase):
    def test_cooperate_cooperate(self):
        self.assertEqual(single_round(COOPERATE, COOPERATE), (3, 3, 'CC'))

    def test_cooperate_defect(self):
        self.assertEqual(single_round(COOPERATE, DEFECT), (0, 5, 'CD'))

    def test_defect_cooperate(self):
        self.assertEqual(single_round(DEFECT, COOPERATE), (5, 0, 'DC'))

    def test_defect_defect(self):
        self.assertEqual(single_round(DEFECT, DEFECT), (1, 1, 'DD'))

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            single_round(2, 0) # Invalid input

if __name__ == '__main__':
    unittest.main()