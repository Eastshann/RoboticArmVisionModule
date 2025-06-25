import unittest
import utils

class TestUtils(unittest.TestCase):
    def test_angle_to_control_value(self):
        self.assertEqual(utils.angle_to_control_value(0), 500)
        self.assertEqual(utils.angle_to_control_value(120), 1000)
        self.assertEqual(utils.angle_to_control_value(-120), 0)

    def test_control_value_to_angle(self):
        self.assertEqual(utils.control_value_to_angle(500), 0)
        self.assertEqual(utils.control_value_to_angle(1000), 120)
        self.assertEqual(utils.control_value_to_angle(0), -120)


if __name__ == '__main__':
    unittest.main()