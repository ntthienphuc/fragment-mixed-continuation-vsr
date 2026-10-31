# Fragment-Mixed Continuation for Cross-Corpus Visual Speech Recognition

Anonymous Pattern Recognition review draft, updated 31 October 2026. The current
manuscript has 35 pages, seven figures and nine tables. All primary recognition
results include completed full populations, including the whole-word short-only
follow-up. Author names and affiliations will be added by the authors later.

## Main evidence

The original VLR-to-ViCocktail comparison improves common-decoder WER by 3.41
and 3.17 percentage points across two fitted seeds. A separate matched seed-17
comparison gives the following whole-utterance results:

| Training views | VLR WER (%) | ViCocktail WER (%) |
| --- | ---: | ---: |
| Full-only | 38.029 | 67.369 |
| Fixed-time mixed | 38.580 | 63.846 |
| Whole-word mixed | 38.606 | 63.388 |
| Whole-word short-only | 44.844 | 70.715 |

Each matched state covers 11,307 VLR clips and 1,167 ViCocktail clips using the
same common readout. Short-only loses 6.238 source and 7.328 destination WER
points relative to whole-word mixing. Increased deletions outweigh fewer
substitutions and insertions. Retaining scheduled full presentations helps in
this fixed-update comparison, but frame/token exposure is unequal. The result
does not establish an optimal mixing ratio or universal failure of short-only
training. The short-only schedule retains 598 full-view fallbacks among 48,000
presentations.

Mixed training still incurs source costs of 0.551 and 0.577 points relative to
full-only. Conventional word-aligned fragments improve destination WER more than
the tested fixed-time rule; the evidence does not support a unique boundary
mechanism. A paired token-normalized control retains a 3.460-point fixed-time
transfer gain. The English LRS2-to-WildVSR setting gives up source performance
without a demonstrated destination gain.

## Artifacts and reproduction

- `Manuscript.pdf`: anonymous reading copy.
- `Overleaf_Source.zip`: complete LaTeX, bibliography and figures. Select
  `main.tex` and pdfLaTeX; Python and shell escape are unnecessary.
- `Data.zip` and `Code.zip`: portable numerical reproduction, including 60
  aggregate rows and 49 conditional paired intervals. See `Code/README.md`.
  Some model identities recur across comparison families.
- `Matched_Training_Source.zip`: six-state matched native/normalized/short-only
  training source, checked against the executed training identities.
- `Research_Code.zip`: full primary-study scientific source with 1,719 frozen
  source/dependency files, including 542 Python files and the upstream reference
  implementation. Its extracted verifier passed 27 CPU tests and reproduced
  all 60 aggregate rows and 49 intervals. See its `REPRODUCE.md` for the mapping
  from paper components to preparation, training, inference and analysis code.
- `Submission/`: five concise highlights, a cover-letter draft and author-completion notes, including
  draft disclosure of the assistance used in research and writing.
- `Submission/Teacher_Brief_VI.md`: a Vietnamese speaking brief, the main matched
  comparison, the completed greedy diagnostic, and the contribution/claim limits.
- `Research_Notes/Next_Decisions.md`: scientific interpretation and scope history.

The numerical command runs on CPU without a research workspace, model weights,
video downloads or network access. The neural training and inference source
requires separately obtained, hash-matched data and checkpoints. These two
levels of reproducibility must not be conflated. The full research-code archive
was freshly extracted, hash-checked, and executed for the CPU checks above;
no fresh 6,000-update neural training replication is implied.

The completed fixed-greedy CTC diagnostic is a separate exploratory supplement.
`Supplement.pdf` reports all six model/population combinations and four paired
intervals. `Greedy_Diagnostic.zip` contains its source, protocol, every saved
CTC path and transcript, and CPU code to reproduce 37,422 clip/model records.
The full-population short-only disadvantage persists against both full-only
and whole-word mixing under this readout. The check reuses the fitted states
and test corpora and does not replace the primary common-decoder results.

## Verification and provenance

The 24 short-only assessment shards and the complete training checkpoint/model
were locally rehashed, restored and independently recounted. Cross-host parity
was checked before the disjoint host-owned shards were pooled; the extra parity
run is excluded from the population totals. All three paid Vast compute
instances were confirmed inactive after backup. Stored data were retained.

The current PDF passed numerical verification, inspection of all 35 rendered
page layouts, and a fresh standalone compilation of its LaTeX ZIP with exact
page-text equality. The four-page supplement also passed visual review,
fresh extracted LaTeX compilation and standalone reproduction of all 37,422
saved clip/model paths, six aggregate rows and four paired intervals. The final
delivery receipt records the archive hashes and full delivered bundle.
No online Overleaf upload, public release or journal submission is implied.

Earlier source snapshots and receipts remain in `Review_*/`. The complete
pre-short-only source is preserved in `Review_09_Short_Only/`. The historical
interval-transcription repository is a different paper; it is not the release
for this continuation study. Third-party data and checkpoint terms remain
with their providers. The layout uses one column, 12 pt, double spacing and
numbered pages. Recheck pagination after adding author metadata.
