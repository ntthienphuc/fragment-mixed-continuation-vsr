# Full-population T2/T4 mechanism extension

This follow-up was prepared after the completed Vietnamese T0/T1 primary
comparison and while English calibration was running. It is not a new hidden
Vietnamese test. No T2/T4 assessment outcome has been used to select its arms.

The question is whether the whole-utterance transfer benefit of mixed full/crop
training depends on missing visual context or on a full-view self-teacher.
Compare the existing final 6,000-update seed-17 states: T0, T1, T2 and T4.
T2 shares T1's cropped student and targets but adds its archived CTC teacher
term. T4 shares the core and targets but gives the encoder surrounding frames.
Their training computations differ; these are not FLOP-matched controls.

Retain both readout comparisons. The primary uses each model's independently
selected decoder from the original eleven-candidate grid. The secondary uses
the fixed native decoder with CTC weight 0.1, beam 40, penalty 0, no external
language model, and a full encoded-frame search cap. All inference is FP32
and reads each original whole clip. The full video is not assembled from crops.

For Vietnamese, use all 768 ViCocktail development clips for calibration and
all 1,167 assessment clips. For English, calibrate on all 1,082 LRS2 development
clips and evaluate all 2,854 released WildVSR clips. These are distinct transfer
regimes. Keep every short clip and the parent normalization/token units.
Do not aggregate Vietnamese whitespace-token and English word-error counts.

Freeze the decoder choices for all four new states before starting any new
assessment. Use only existing development data for that choice. The anchor
is fixed independently of target performance; never report the better of the
primary and anchor columns. Reuse T0/T1 and practical-reference predictions
only after the complete parent results and exact identities are verified.

The new plan has 64 tasks: two states per language, two stages and eight
shards. Each new state covers the complete parent cohort exactly once.
English and assessment partitions retain the parent memberships; Vietnamese
development uses all 768 positions sorted by length and distributed over eight
shards. Parent T1's 237-clip incremental queue cannot be reused for a new state. At most
8,042 new assessment model-clip instances require recognition, each with the
selected decoder and the fixed anchor when they differ. Primary reference
models are context, not matched training controls.

Report complete-population WER and substitutions/deletions/insertions. Retain
all T2/T4 outcomes, including negative ones. Vietnamese uses the same 20 video
source groups and 10,000 paired bootstrap draws as its parent, conditional on
these trained states. English has no source grouping in the released local
metadata; any clip bootstrap must retain that limitation. Only seed 17 exists
for T2/T4. This comparison cannot establish seed-population robustness.

Before target recognition, require strict model loading, native source replay
on the execution device, data/hash checks, a frozen worker and independent
verifier, all decoder selections, and a finite admission with backup. Preparation
may validate the four models on CPU using fixed development clips. It must not
add work to the active pod, alter its guard or allocate another server.

The next training question remains open even if T2/T4 is informative: compare
conventional word-aligned augmentation and explicitly controlled loss/token
exposure. Existing crop augmentation and self-distillation alone do not establish
novelty. The possible contribution is a tested explanation and useful method
for whole-utterance recognition under corpus shift.
