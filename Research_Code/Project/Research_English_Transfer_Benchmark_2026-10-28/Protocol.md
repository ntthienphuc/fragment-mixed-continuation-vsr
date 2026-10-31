# Full-clip transfer from LRS2 adaptation to WildVSR

## Research question

Does alternating full-clip and two-second crop supervision during adaptation
improve whole-utterance recognition on a different corpus? Evaluate the existing
English checkpoints without additional training, sliding windows, interval
projection or reference-based correction. The comparison tests transfer of the
training recipe; temporal cropping itself is established prior work.

The matched states are the actual LRS3 initializer EN_B0 and the LRS2 T0/T1
checkpoints at exactly 6,000 updates, seed 17. T0 uses whole clips; T1 alternates
whole clips and crops. Their source clips and update counts match, but frame and
label exposure do not. EN_B1 is a separately trained stronger practical reference,
not the shared initializer. Report all four states regardless of outcome.

## Complete populations and fixed decoder choices

Calibrate each model on all 1,082 prepared LRS2 development clips, including the
903 clips below 82 frames. These are the original source-validation videos;
their accuracy is a development result, not an independent source-domain test.
The 81 known video-source IDs are retained without relabeling them as speakers.

Choose one decoder for each model by minimum aggregate word edit errors over
the complete source development set. Use this fixed tie order: native beam
CTC weights 0.1, 0.5, 0.9, 1.0; then greedy blank biases 0, -0.5, +0.5, -1, +1,
-2, +2. Beam size is 40, insertion penalty is zero, and search uses the complete
encoded frame count. No external language model is used. Freeze all four
decoders together before any WildVSR recognition. Do not select checkpoints,
hyperparameters, clips or report columns from destination outcomes.

Assess every one of the 2,854 released WildVSR clips. Its author-path grayscale
pixels, 25fps convention and original transcripts are already verified. Keep
the lossless uint8 input and apply the native center-crop/normalization once.
English uses 5,049 token classes, blank 0 and EOS 5,048; all execution is FP32.
The source feature cache is lossless FP32 and must pass identity and fresh-pixel
replay before reuse. Cache use does not reduce the evaluation population.

The primary readout is each model's independently source-selected decoder.
Also report the prespecified common anchor: CTC weight 0.1, beam 40, penalty 0,
complete frame cap. Its definition is bound to the cross-corpus anchor protocol
sealed before expanded ViCocktail assessment. If the primary equals the anchor,
reuse the identical verified prediction. Otherwise produce exactly two fixed
readouts. Never choose the better column after looking at WildVSR.

## Outcomes and interpretation

Primary outcomes are full-population WER, S/D/I, and paired T0 minus T1 WER
percentage points. Use unchanged NFC/lowercase/whitespace normalization; the
released references contain 48,037 words under that convention. Keep the source
development table separate. Do not pool English words and Vietnamese whitespace
units, or attribute differences between corpus pairs to language alone.

The archive lacks the source-video and speaker mapping described in the dataset
paper. Do not invent clusters from numeric filenames. Report paired clip
bootstrap intervals, if calculated, only as conditional descriptive intervals
over the released clips, using 10,000 draws and seed 20261028. They do not
establish uncertainty over independent video sources. Both readouts use the same
resampled clip weights. Length strata below/at least 82 frames are secondary
diagnostics, not alternative success endpoints.

A positive primary T0/T1 difference supports further mechanism controls; it
does not establish a novel algorithm. Inspect the common anchor and source
tradeoff as well. If the full benchmark contradicts the Vietnamese development
signal, retain that result and narrow the transfer claim. One English training
seed cannot establish training-seed robustness. Checkpoint lineage also does
not prove exclusion of every upstream pretraining overlap.

## Execution and completion

Each task must have complete coverage, re-counted edit errors, matching assets,
and fresh-pixel replay on deterministic first/last task members. All 32 source
development tasks are required before decoder selection; all 32 assessment tasks
are required before a complete result. Preserve incomplete attempts separately.

Local work may be admitted with an explicit deadline and memory gate. Cloud
execution requires a measured ETA, a separate finite allocation below USD1/hour,
at least USD1 reserve, verified backup and provider stop handling. Preparing this
protocol creates no paid allocation and changes no active ViCocktail job.
