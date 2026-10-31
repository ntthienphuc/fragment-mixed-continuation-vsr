import csv
import tempfile
import unittest
from pathlib import Path
from phase5.importers import vlr_records, lrs2_label
from phase5.views import window_start, span
from phase5.prepare_worker import feasible
from phase5.text import reference_text


class Annotations(unittest.TestCase):
    def test_english_unknown_surfaces_use_checkpoint_unknown_unit(self):
        from phase5.text import Tokenizer
        from phase5.common import ROOT
        tokens = Tokenizer(ROOT/'assets/tokenizers/en.model', ROOT/'assets/tokenizers/en_units.txt', 'en')
        text = '846'
        native = tokens.processor.encode(text.upper(), out_type=int)
        expected = [tokens.mapping[tokens.processor.id_to_piece(i)] for i in native]
        self.assertEqual(tokens.encode(text), expected)
        self.assertIn(tokens.mapping['<unk>'], expected)

    def test_reference_matches_language_tokenizer_policy(self):
        self.assertEqual(reference_text('Chào, bạn! Việt-Nam', 'vi'), 'chào bạn việt nam')
        self.assertEqual(reference_text("WE'RE COOKING", 'en'), "we're cooking")

    def test_two_second_windows_have_matched_context(self):
        for length in [82, 100, 154, 400]:
            start = window_start(length, 'identity', core=50)
            for mode, expected in [('base', 50), ('past', 66), ('both', 82)]:
                left, right, begin, end = span(length, start, mode, core=50)
                self.assertEqual(right-left, expected)
                self.assertEqual(end-begin, 50)
                self.assertEqual(left+begin, start)
        with self.assertRaises(ValueError):
            window_start(81, 'identity', core=50)

    def test_punctuation_only_window_is_not_a_preparation_failure(self):
        class Tokens:
            def encode(self, text):
                if text == '...':
                    raise ValueError('Empty target')
                return [1, 2]
        self.assertFalse(feasible(Tokens(), '...', 50))
        self.assertTrue(feasible(Tokens(), 'word', 50))

    def test_lrs2_numeric_transcript_has_no_word_times(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'label.txt'
            path.write_text('Text: 70 100 IS THE ESTIMATE\nConf: 4\n')
            text, words = lrs2_label(path)
            self.assertEqual(text, '70 100 IS THE ESTIMATE')
            self.assertEqual(words, [])

    def test_blank_zero_row_preserves_reference(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root/'manifests').mkdir()
            fields = ['split', 'clip_id', 'source_id', 'label_path', 'video_path',
                      'transcript', 'video_blake2b_128']
            with (root/'manifests/val_strict.csv').open('w', newline='', encoding='utf-8') as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow(dict(zip(fields, ['val', 'a', 's', 'label.csv', 'a.mp4',
                                                   'ờ ờ', 'unused'])))
            label = root/'label.csv'
            label.write_text('Word,Start,End\n"ờ,",0,1\n ,1,1\nờ,1,2\n', encoding='utf-8')
            row = next(vlr_records(root, 'val'))
            self.assertEqual(row['ignored_blank_zero_rows'], 1)
            self.assertEqual(len(row['words']), 2)
            label.write_text('Word,Start,End\nờ,0,1\n ,1,1.5\nờ,1.5,2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                next(vlr_records(root, 'val'))
            label.write_text('Word,Start,End\nờ,0,1\n ,1,1\nkhác,1,2\n', encoding='utf-8')
            with self.assertRaises(ValueError):
                next(vlr_records(root, 'val'))
