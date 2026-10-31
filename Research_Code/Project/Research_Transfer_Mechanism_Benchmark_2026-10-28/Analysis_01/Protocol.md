# Complete-population mechanism analysis

This analysis is specified after the completed T0/T1 Vietnamese and English
comparisons and before any new T2/T4 assessment prediction. It concerns the
four existing final-step seed-17 T2/T4 states, not checkpoint selection.

The main new contrasts are T1 minus T2 and T1 minus T4 within each language.
Positive WER differences favor T2 or T4. Retain T0 minus T1 as a parent-result
check and T0 minus T2/T4 as supporting continuation comparisons. Include B0
and the stronger practical references, while distinguishing their training
resources. Do not pool English words with Vietnamese whitespace units.

The primary readout is independently selected on the full development cohort
using the already locked candidate grid. The secondary readout is beam_c0.1.
Report both even when they disagree. All four selections must precede every
new assessment prediction. Do not choose the better test readout or suppress
a negative method/language result.

Use all 1,167 ViCocktail clips and all 2,854 WildVSR clips. Vietnamese paired
bootstrap resamples its 20 video sources with 10,000 draws, seed 20261027.
English uses the parent's released-clip bootstrap with 10,000 draws, seed
20261028; independent source grouping is unavailable. Intervals are pointwise,
unadjusted for multiple comparisons, and conditional on the fitted states.
They are not uncertainty estimates over training seeds. A favorable isolated
interval will not alone establish a new method or determine a winner.

Report corpus WER, S/D/I counts, total output units and paired contrasts. The
parent controls are joined by exact clip identity, frame count and reference
denominator, not by row count alone. Require all 32 new assessment tasks and
the complete source selection evidence; incomplete matrices are not summarized.
No result is automatically added to the interval-transcription manuscript.

The independent checks reproduce four completed parent WER/CI comparisons
from 42,544 CSV records. Synthetic equality and corrupted-table tests validate
the arithmetic and coverage checks. These tests do not substitute for actual
worker/verifier integration or the eventual complete T2/T4 evaluation.
