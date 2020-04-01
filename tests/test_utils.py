import unittest
import torch

from SSD import utils


class TestUtils(unittest.TestCase):
    def test_toPoint(self):
        inputs = torch.Tensor([[1, 1, 2, 2], [2, 2, 4, 4]])
        result = [[0, 0, 2, 2], [0, 0, 4, 4]]
        self.assertEqual(utils.toPoint(inputs).tolist(), result)

    def test_toCenter(self):
        inputs = torch.Tensor([[0, 0, 2, 2], [0, 0, 4, 4]])
        result = [[1, 1, 2, 2], [2, 2, 4, 4]]
        self.assertEqual(utils.toCenter(inputs).tolist(), result)

    def test_intersection(self):
        box_a = torch.Tensor([[0, 0, 2, 2], [0, 0, 4, 4]])
        box_b = torch.Tensor([[1, 1, 3, 3], [1, 1, 2, 2]])

        inter = utils.intersection(box_a, box_b)
        self.assertEqual(inter.tolist(), [[1, 1], [4, 1]])

    def test_jaccard(self):
        box_a = torch.Tensor([[0, 0, 2, 2], [0, 0, 4, 4]])
        box_b = torch.Tensor([[1, 1, 3, 3], [1, 1, 2, 2]])

        score = utils.jaccard(box_a, box_b)
        self.assertEqual(score.tolist(), [[0.1428571492433548, 0.25], [0.25, 0.0625]])

    def test_match(self):
        # default_boxes = torch.Tensor(
        #     []
        # )
        pass
