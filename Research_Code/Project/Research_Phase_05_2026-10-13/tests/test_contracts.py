import copy
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace as S
import numpy as np
import torch
from phase5.text import score, Tokenizer, required_frames
from phase5.common import ROOT
from phase5.views import span, target_text, window_start
from phase5.readout import BoundaryReadout
from phase5.detector import largest_detection
from phase5.transfer import extract
from phase5.compare import compare


class Contracts(unittest.TestCase):
    def test_metrics(self):
        self.assertEqual(score('a b c', 'a x c d')['word']['errors'], 2)
        self.assertEqual(score('xin chào', '')['word']['deletions'], 2)
        self.assertEqual(score('chào', 'chào')['word']['errors'], 0)
        self.assertEqual(required_frames([1, 1, 2, 2]), 6)

    def test_tokenizers(self):
        for language, text in [('en', 'HELLO WORLD'), ('vi', 'xin chào')]:
            t = Tokenizer(ROOT / f'assets/tokenizers/{language}.model',
                          ROOT / f'assets/tokenizers/{language}_units.txt', language)
            self.assertEqual(t.decode(t.encode(text)), text)

    def test_geometry(self):
        for n in [157, 200, 400]:
            start = window_start(n, 'sample')
            for mode, length in [('base', 125), ('past', 141), ('both', 157)]:
                left, right, a, b = span(n, start, mode)
                self.assertEqual(right-left, length)
                self.assertEqual(b-a, 125)
                self.assertEqual(left+a, start)
        words = [{'text': 'left', 'start': 0, 'end': 1},
                 {'text': 'zero', 'start': 1, 'end': 1},
                 {'text': 'right', 'start': 2, 'end': 3}]
        self.assertEqual(target_text(words, 25, 50), 'zero')

    def test_readout_identity_and_learning(self):
        r = BoundaryReadout(16, 4)
        x = torch.randn(1, 9, 16, requires_grad=True)
        self.assertTrue(torch.equal(r(x, 2, 7), x[:, 2:7]))
        optimizer = torch.optim.SGD(r.parameters(), lr=.01)
        for _ in range(2):
            optimizer.zero_grad()
            r(x, 2, 7).square().mean().backward()
            optimizer.step()
        self.assertGreater(r.penalties.grad.abs().sum().item(), 0)
        self.assertFalse(torch.equal(r(x, 2, 7), r(x, 2, 7, boundary=False)))

    def test_face_area(self):
        def face(w, h):
            return S(location_data=S(relative_bounding_box=S(width=w, height=h)))
        faces = [face(.8, .8), face(.2, .2)]
        self.assertIs(largest_detection(faces), faces[0])

    def test_archive_escape(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../escape', 'bad')
            with self.assertRaises(ValueError):
                extract(archive, Path(tmp) / 'out')

    def test_paired_statistics(self):
        left = [{'clip_id': str(i), 'source_id': str(i), 'reference': 'a b',
                 'view': 'window', 'context': 'both', 'decoder': 'greedy',
                 'scores': score('a b', 'a')} for i in range(4)]
        right = copy.deepcopy(left)
        for row in right:
            row['scores'] = score('a b', 'a b')
        self.assertEqual(compare(left, right, 100)['gain_pp'], 50)
        right[0]['reference'] = 'changed'
        with self.assertRaises(ValueError):
            compare(left, right, 100)


if __name__ == '__main__':
    unittest.main()
