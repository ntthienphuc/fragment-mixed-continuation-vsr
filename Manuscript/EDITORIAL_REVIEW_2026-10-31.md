# Final editorial review, 31 October 2026

The anonymous manuscript remains 35 pages, with seven figures and nine tables.
This revision changes figure presentation, placement and editorial emphasis.
It does not change measured results, statistical intervals or the scope of the study.

## Figures and layout

- Figures 1 and 2 use Latin Modern Roman, matching the manuscript text family.
- Figure 1 uses aligned comparison rows and a separate whole-utterance inference path.
  It is placed after the completed whole-word paragraph and before Section 3.4.
  It no longer interrupts a sentence between pages 8 and 9.
- Figure 2 uses aligned corpus labels, consistent time labels and separated corpus pairs.
  All 24 embedded mouth images retain identical decoded pixels and ordering.
- The five numerical plots and all nine table sources are byte-identical to the prior release.
- Editable TikZ sources, unchanged display images and a two-pass builder are in `Figure_Source/`.
  The Overleaf ZIP includes them under `figure_source/`; main compilation uses prebuilt PDFs.
- All 35 manuscript pages were rendered and reviewed, including the two figure pages.

## Writing

The Introduction and Conclusion foreground the contribution of matched schedule,
decoder and exposure comparisons. The Discussion explains the short-only result
before its interpretation boundary and consolidates repeated qualifications.
The cover letter leads with the recognition problem, measured contribution and
evidence supporting the proposed continuation strategy.

The text retains the distinctions needed to interpret the evidence: the short-only
control uses seed 17; frame/token exposure is unequal; assessment sets were reused;
bootstrap intervals condition on fitted states; and English transfer does not show
the Vietnamese gain. Prior mixed-length training remains acknowledged. No claim
of an optimal mixing ratio, universal transfer benefit or new recognizer architecture
has been added.

## Numerical and supplementary audit

The frozen research package passed source verification for 1,719 files, including
542 Python files, six executed matched training identities and a 101-file reference
dependency graph. CPU reproduction recovered 60 primary aggregate rows and
49 paired intervals. The fixed greedy-CTC diagnostic recovered 37,422 saved
clip/model records, six aggregate rows and four paired intervals.

The main short-only contrast was independently checked from saved edit counts:

| Population | Additional errors vs. whole-word mixed | Reference units | WER cost (pp) |
| --- | ---: | ---: | ---: |
| VLR | 14,096 | 225,960 | 6.238272 |
| ViCocktail | 2,205 | 30,091 | 7.327772 |

The four-page `Supplement.pdf` is identical to the diagnostic package's reading
copy. Clean compilation of its LaTeX source reproduced every page's text exactly.
The main Discussion points to supplementary Tables 1 and 4 for WER and paired
contrasts. Supplementary evidence and code are present at the following paths:

| Material | Repository path |
| --- | --- |
| Reading copy | `Manuscript/Supplement.pdf` |
| Protocol and typeset tables | `Greedy_Diagnostic/Source/` |
| Full-precision six-state aggregates and contrasts | `Greedy_Diagnostic/Results/` |
| Saved CTC paths, transcripts and identities | `Greedy_Diagnostic/Data/Collected/` |
| Portable diagnostic recount | `verify_greedy.py` and `Greedy_Diagnostic/Code/` |
| Primary measurements and recount | `Data/` and `Code/` |
| Preparation, training and inference source | `Research_Code/` and `Matched_Training_Source/` |

These checks reproduce saved measurements and predictions; they do not rerun
neural training or inference. Author metadata and submission declarations remain
for the authors to complete. No journal submission was made during this review.
