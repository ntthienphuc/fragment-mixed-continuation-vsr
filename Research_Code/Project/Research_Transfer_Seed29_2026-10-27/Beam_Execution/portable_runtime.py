"""Exact native decoder/head exports; no encoder or learned approximation."""

import sys

from common import ENGINE, ROOT, STORE, VENDOR, plan, read, sha

sys.path.insert(0, str(VENDOR))
sys.path.insert(0, str(ENGINE))
import torch
from torch import nn
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.decoder.transformer_decoder import TransformerDecoder
from espnet.nets.scorers.ctc import CTCPrefixScorer
from native_runtime import Runtime


class NativeReadout(nn.Module):
    def __init__(self, payload):
        super().__init__()
        self.decoder = TransformerDecoder(odim=2002, attention_dim=768, attention_heads=12,
                                          linear_units=3072, num_blocks=6)
        self.decoder.load_state_dict(payload["decoder"], strict=True)
        self.ctc = CTC(2002, 768, 0.1, reduce=True)
        self.ctc.load_state_dict(payload["ctc"], strict=True)
        self.odim = 2002
        self.blank = 0
        self.sos = self.eos = 2001
        self.eval().float().requires_grad_(False)

    def scorers(self):
        return {"decoder": self.decoder, "ctc": CTCPrefixScorer(self.ctc, self.eos)}


class DecoderRuntime(Runtime):
    def __init__(self, model_id, device="cuda"):
        frozen = plan()
        receipt = read(STORE / "Models" / (model_id + ".json"))
        path = STORE / "Models" / (model_id + ".pt")
        assert sha(path) == receipt["export_sha256"]
        assert receipt["checkpoint_sha256"] == frozen["models"][model_id]["entry"]["checkpoint_sha256"]
        payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        self.root = ROOT
        self.registry = read(ENGINE / "model_registry.json")
        assert sha(ENGINE / "model_registry.json") == frozen["registry_sha256"]
        self.factory_path = ROOT / self.registry["factory_source"]
        self.model_id = model_id
        self.device = torch.device(device)
        self.model = NativeReadout(payload).to(self.device)
        self.device = next(self.model.parameters()).device
        self.vocabulary = payload["vocabulary"]
        self.blank_id = 0
        self.eos_id = 2001
        self._load_search()
        self.identity = {"model_id": model_id, "export_sha256": receipt["export_sha256"],
                         "checkpoint_sha256": receipt["checkpoint_sha256"], "device": str(self.device),
                         "precision": "float32", "torch": str(torch.__version__), "exact_native_readout": True}


def clear_attention(model):
    for module in model.modules():
        if hasattr(module, "attn"):
            module.attn = None
