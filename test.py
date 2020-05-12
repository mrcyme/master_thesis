from samplers import *
import json
import unittest
config = json.load(open("config.json",'r+'))


class TestSamplers(unittest.TestCase):
    def test_sobol(self):
        s = sobol_sampling(config,10)
        self.assertEqual(s.shape,(len(config),10))

    def test_lhs(self):
        s = lhs_sampling(config,10)
        self.assertEqual(s.shape,(len(config),10))

    def test_random(self):
        s = random_sampling(config,10)
        self.assertEqual(s.shape,(len(config),10))



if __name__ == '__main__':
    unittest.main()
