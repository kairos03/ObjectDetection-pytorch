import unittest
import torch

from SSD import models

class TestModel(unittest.TestCase):
    def test_model(self):
        model = models.ssd.vgg_ssd()
        model = model.train()

        inputs = torch.Tensor(5, 3, 300, 300)
        outputs = model(inputs)        

        model.check_base()
