import unittest
import torch
from torch import nn
from phase5.spatial_pool import SpatialPool, replace_spatial_pool


class SpatialPoolingTests(unittest.TestCase):
    def test_native_outputs_and_gradients_with_ties(self):
        for ties in [False,True]:
            torch.manual_seed(17)
            raw = torch.randn(2,4,7,18,18)
            if ties:
                raw = raw.round()
            a = raw.clone().requires_grad_()
            b = raw.clone().requires_grad_()
            expected = nn.functional.max_pool3d(a,(1,3,3),(1,2,2),(0,1,1))
            actual = SpatialPool()(b)
            upstream = torch.randn_like(expected)
            expected.backward(upstream)
            actual.backward(upstream)
            self.assertTrue(torch.equal(expected,actual))
            self.assertTrue(torch.equal(a.grad,b.grad))

    def test_replacement_preserves_checkpoint_state_and_rejects_other_geometry(self):
        frontend = nn.Module()
        frontend.frontend3D = nn.Sequential(nn.Identity(),nn.Identity(),nn.Identity(),
            nn.MaxPool3d((1,3,3),(1,2,2),(0,1,1)))
        before = frontend.state_dict()
        replace_spatial_pool(frontend)
        self.assertEqual(before,frontend.state_dict())
        frontend.frontend3D[3] = nn.MaxPool3d((3,3,3))
        with self.assertRaises(ValueError):
            replace_spatial_pool(frontend)
