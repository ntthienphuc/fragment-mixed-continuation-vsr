# Full-clip ViCocktail transfer confirmation

Prepared while the seed-29 development replication is running. Preparation does
not authorize another paid job or replace the active server's budget deadline.

## Question and models

Test whether alternating full/cropped supervision improves whole-clip recognition
on the complete official ViCocktail test after independent decoder calibration
on all available development clips. No sliding, target-word projection or
reference-based inference is used.

The matched family is the common VLR initializer B0 and final-step T0/T1 models
from both seeds 17 and 29. All four adaptations use step 6,000; selected-best
checkpoints are not candidates. Retain both seeds regardless of direction.
Include the available native ViCocktail model as a practical target-trained
reference, using its own verified implementation/preprocessing. It is not a
matched training control and will not be merged into the paired seed analysis.
Its execution path must pass native replay before inference admission.

## Data and decoder selection

Use all 768 development clips and all 1,167 official test clips, including clips
shorter than 82 frames and those without word timing. The existing 531-clip
development comparison remains separately reported with its original scope.
Development/test IDs, source IDs and video hashes must be disjoint. Preserve
every original normalized transcript and each input video; never replace labels
or remove samples after seeing predictions. Missing/unreadable inputs fail the
complete benchmark until repaired or explicitly reported as incomplete.

For each model, choose one decoder by minimum aggregate edit errors on all 768
development clips. The candidate order is frozen before reading missing-clip
results: beam CTC weights 0.1, 0.5, 0.9, 1.0, followed by greedy blank biases
0, -0.5, +0.5, -1, +1, -2, +2. Use beam size 40 and zero insertion penalty,
matching the existing native decoder. Ties use this exact order. Do not select
decoders from the 531-clip subset alone or from test results. Store a selection
lock binding every development prediction, checkpoint and inference setting.

The test worker accepts only that locked decoder per model. Additional test
decoder searches are not required. The 531 development predictions may be reused
only when checkpoint, raw decoded pixels, transforms, precision, vocabulary,
decoder settings and complete verification receipts agree. Five matched models
need 237 additional development clips each; the native reference needs all 768.
Reuse reduces duplicate compute, not the evaluation population.

## Outcomes and inference limits

Primary results are whole-clip Vietnamese whitespace-token WER, S/D/I and paired
T0 minus T1 differences for each fixed seed. Keep the customary WER name while
specifying that Vietnamese space-separated units are often syllables. Report
both seeds and their mean as a fixed two-seed summary; do not treat two seeds as
a precise estimate of the population distribution over random training seeds.

Use paired bootstrap over the 20 original video-source groups, 10,000 draws and
seed 20261027. Use the same sampled source weights for both seeds. This estimates
uncertainty over those source groups conditional on the two fitted models and
development-selected decoders, not uncertainty over new training seeds. Also
report unweighted per-source differences and duration strata (<82 and >=82 frames).
Those strata are secondary diagnostics; the entire test is the primary endpoint.
Measure inference time, peak memory and number of forward/search calls separately
from export/upload/verification overhead. Do not infer speed or FLOPs from the
number of training frames.

The official test has already appeared elsewhere in this project. This is a
frozen benchmark extension, not a newly hidden or independently collected test.
Source groups are video identifiers, not independently verified speaker IDs.
Checkpoint metadata establishes declared training lineage; it does not by itself
prove absence of upstream pretraining overlap. Native-reference data exposure
and model preprocessing must remain explicit in comparisons.

## Admission and decision

Before new GPU work: require the complete seed-29 receipt and interpretation,
current input hashes, native runtime checks, a bounded queue, measured resource
estimate and verified backup/stop handling. A positive full benchmark would still
need exposure/loss controls and matched source-domain comparisons before a causal
claim about local supervision. A disagreement between seeds or failure on the
full benchmark must narrow the claim and guide the next explanation, rather than
trigger selection of a favorable subset or a different checkpoint.

This branch may support a separate paper on full-clip transfer. It will enter the
existing interval-transcription paper only if it directly answers that paper's
question. No manuscript changes are made by preparing this protocol.
