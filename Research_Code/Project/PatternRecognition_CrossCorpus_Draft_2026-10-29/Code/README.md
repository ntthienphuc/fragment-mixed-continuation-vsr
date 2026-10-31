# Numerical reproduction

This small package recomputes whole-corpus WER from the supplied per-clip edit
counts and reproduces the paired bootstrap intervals. It runs without a GPU,
model weights, video downloads, a research workspace, or network access.

## Run

Extract Code.zip and Data.zip next to one another so that the directories are
`Code/` and `Data/`, then run from their parent directory:

```text
python -m pip install -r Code/requirements.txt
python Code/reproduce.py --data Data --output Recomputed
```

The output directory must not already exist. The script exits with an error if
clips are duplicated, paired references differ, edit counts are inconsistent,
a reported WER differs, or a supported confidence interval cannot be reproduced.
It does not overwrite the source data or an earlier reproduction.

Verified with Python 3.11.9 and NumPy 2.0.2. The completed export reproduces 60 WER
rows and 49 paired intervals, each with 10,000 resamples. `Data/Matched_Source/`
contains the complete new crop-control source matrix and is checked by the
same command. Original and newly fitted model identities remain separate.

## What is reproduced

- Original VLR/ViCocktail and LRS2/WildVSR destination aggregate rows and the six
  FC-versus-FM intervals in the main comparison.
- Original source aggregate rows and twelve source comparison intervals.
- Sixteen same-target context-control comparisons, including the negative and
  inconclusive outcomes and their original resampling units.
- The new native full/fixed-time/whole-word source and destination comparisons.
- The four-state native-versus-token-normalized loss comparison and interaction.
- The complete four-state full/fixed-time/whole-word/short-only source and
  destination comparison, including four predeclared short-only intervals.

Rows repeated across comparison families retain their original identities;
60 exported rows do not mean 60 independently trained models. The separate
exploratory greedy readout follow-up has its own protocol and reproduction.

The historical Vietnamese destination comparison uses multinomial resampling
of 20 recording groups with seed 20261027. Historical English destination
resamples released clips with seed 20261028. Source and crop comparisons use
the seeds stored with their paired comparisons. The loss interaction uses the
same group draws across all four states. Each interval is conditional on the
fitted states and the observed assessment population, without multiplicity
adjustment; the script does not create an independent confirmation.

## Scope

These scripts reproduce statistics from exported edit counts. They do not rerun
neural training, inference, or transcript alignment. The manuscript and model
identities define those experimental protocols. The original study's verified
inference receipts remain separate from this portable numerical package.
The existing public interval-transcription repository is a different paper;
this package does not claim that it contains these continuation experiments.

Original code here is MIT licensed. Derived numerical tables are provided under
CC BY 4.0. Third-party videos and model checkpoints are not included or relicensed.
