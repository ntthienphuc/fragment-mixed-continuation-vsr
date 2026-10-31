"""Preserve native Auto-AVSR reduction/dropout and gradient behavior on CPU."""
import sys
import unittest
import torch
from phase5.common import ROOT
from phase5.ctc_loss import cpu_ctc
sys.path.insert(0,str(ROOT/'vendor/auto_avsr'))
from espnet.nets.pytorch_backend.ctc import CTC


class CTCLossTests(unittest.TestCase):
    def test_matches_native_loss_and_gradients_with_dropout(self):
        torch.manual_seed(61)
        module = CTC(5,3,.1)
        module.train()
        frames = torch.randn(2,8,3,requires_grad=True)
        targets = torch.tensor([[1,2,2],[3,2,-1]])
        lengths = torch.tensor([8,6])
        state = torch.get_rng_state()
        native,_ = module(frames,lengths,targets)
        native.backward()
        expected_frames = frames.grad.clone()
        expected_weight = module.ctc_lo.weight.grad.clone()
        module.zero_grad()
        frames.grad = None
        torch.set_rng_state(state)
        logits = module.ctc_lo(module.dropout(frames)).transpose(0,1)
        packed = torch.cat([row[row!=-1] for row in targets])
        actual = cpu_ctc(logits.log_softmax(2),packed,lengths,[3,2])/2
        actual.backward()
        self.assertTrue(torch.equal(actual,native))
        self.assertTrue(torch.equal(frames.grad,expected_frames))
        self.assertTrue(torch.equal(module.ctc_lo.weight.grad,expected_weight))
