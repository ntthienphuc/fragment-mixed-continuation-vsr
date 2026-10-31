# Fragment-Mixed Continuation for Cross-Corpus Visual Speech Recognition

Anonymous manuscript, executed research source, derived measurements and
reproduction scripts. This repository is separate from the interval-transcription
paper and from the LipTrace application.

[Main paper](Manuscript/Manuscript.pdf) · [Supplement](Manuscript/Supplement.pdf) ·
[LaTeX source](Manuscript/Source/) · [Overleaf ZIP](Manuscript/Overleaf_Source.zip) ·
[Training and inference map](Research_Code/REPRODUCE.md)

## Scope

The paper studies source-side fragment mixing during visual-speech continuation
and evaluates source retention and cross-corpus transfer. It includes the original
FC/FM/FM-ST/CP cases, matched full/fixed-time/whole-word/short-only controls,
component-wise token-normalized training and a fixed-greedy CTC diagnostic.

The main paper is 35 pages. The exploratory supplement is four pages and reuses
the fitted states and test corpora. It is not an additional independent training
replication or a new blind test collection. Paper statements and scope limits are
preserved from the reviewed anonymous release of 31 October 2026.

## Reproduce the reported numbers

On Windows, enable long paths for this checkout because the preserved scientific
dependency tree is deep:

```sh
git -c core.longpaths=true clone https://github.com/ntthienphuc/fragment-mixed-continuation-vsr.git
```

Use Python 3.11 or newer. The numerical environment was verified with Python
3.11.9 and NumPy 2.0.2 on Windows. From the repository root:

```sh
python -m pip install -r requirements.txt
python verify.py --output Verification_Run_01
```

The output directory must be new. This checks the repository payload, all 1,719
research source/dependency files, the original training identity, six matched
driver identities and the 101-file native-reference graph. It then reproduces
60 primary aggregate rows and 49 paired intervals, followed by 37,422 saved
model/clip paths, transcripts and edit counts, six diagnostic aggregates and
four diagnostic intervals. It needs no GPU, model weights or video downloads.

Frozen source and data bytes are hash-checked without normalization. For regenerated
diagnostic text exports, the Git verifier permits only CRLF versus LF line-ending
differences across Windows and Linux. Every numerical digit and record must still
match exactly. The original Windows release verifier is preserved unchanged.

The root verifier can also run the 27 original CPU tests with a separate Python
interpreter containing Torch, NumPy, SentencePiece and preparation dependencies:

```sh
python verify.py --output Verification_Run_02 --test-python /path/to/python
```

The observed unit-test environment used Python 3.9.13 and Torch 2.8.0+cpu.
Numerical reproduction and unit checks do not rerun neural training or inference.

## Contents and neural reproduction

| Directory | Purpose |
| --- | --- |
| `Manuscript/` | Reviewed PDFs, LaTeX and Overleaf source |
| `Code/` | Portable primary numerical analysis |
| `Data/` | Frozen derived primary measurements and provenance |
| `Research_Code/` | Original preparation, training, inference and analysis sources |
| `Matched_Training_Source/` | Compact matched-driver packet and identities |
| `Greedy_Diagnostic/` | Saved paths, predictions, counts, analysis and supplement source |
| `verification/` | Git payload manifest and source-package provenance |

`Research_Code/Project/` preserves dated relative paths and byte identities.
Follow [REPRODUCE.md](Research_Code/REPRODUCE.md) for actual neural entry points,
data preparation, native loss variants and environment requirements. Acquire
hash-matched media and checkpoints separately, under their original provider
terms. [External_Assets.json](Research_Code/External_Assets.json) records the
external prerequisites. Historical finite execution admissions are expired and
must not be bypassed; any new run needs its own verified inputs and admission.

## Licenses and manuscript status

The portable numerical code is under its [MIT license](Code/LICENSE). Derived
measurements have [CC BY 4.0 terms](Data/LICENSE-DATA-CC-BY-4.0.txt). Scientific
source and upstream components retain their existing notices and provenance.
There is no blanket relicensing of third-party source, recordings or weights.

The manuscript is anonymous. Author information and journal submission metadata
remain to be completed. This repository does not establish acceptance, a journal
DOI, or an archival software DOI. The previously reviewed manuscript source is
unchanged; its release README records the historical delivery state before this
Git synchronization.
