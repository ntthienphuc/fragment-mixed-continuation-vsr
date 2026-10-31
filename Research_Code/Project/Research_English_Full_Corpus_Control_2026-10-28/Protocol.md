# English full-training-split reference

F0 is an archived Auto-AVSR continuation that uses all 45,838 full-eligible
LRS2 training clips, including short clips. T0/T1 in the current matched
benchmark use the same 7,291 window-eligible clips. This follow-up asks whether
ordinary broader training coverage provides a stronger practical transfer
reference than the mixed full/crop recipe. It was specified after the user
raised the short-example/data-augmentation explanation, before any new F0
WildVSR inference. It is not a new training method.

Use the original final 6,000-update seed-17 state. Its training architecture is
T0; F0 is the public name for the full-corpus training population. The selected
endpoint is fixed, with no new checkpoint choice. The initializer, learning
rate, accumulation and number of updates match the archived English T0/T1
settings, but the number of distinct training clips and exposure distribution
do not. The old training validation view also differs; this evaluation uses
the final state, not a comparison of differently selected best checkpoints.

Calibrate the same eleven candidates on all 1,082 LRS2 development clips:
native beams at CTC weights 0.1/0.5/0.9/1.0, beam 40, penalty 0, and greedy
blank biases 0/-0.5/0.5/-1/1/-2/2 in the parent's frozen order. Minimize pooled
development edit errors, using that order for ties. Evaluate every one of the
2,854 WildVSR clips with the selected decoder and the fixed native 0.1 decoder;
deduplicate recognition only when both settings are exactly the same.

All inference is FP32, without an external language model. Read the original
whole clip once per model; use the complete encoded-frame search horizon and
the same author-path gray96 pixels, tokenizer and normalization as the parent.
No sliding, temporal projection or reference transcript is used for inference.
Original short clips remain in both cohorts. Report WER and S/D/I alongside
the complete matched T0/T1 result and the existing practical B1 reference.

F0 is a practical different-data comparison. If it outperforms T1, that does
not erase the same-population T0/T1 effect, but it limits a claim about the best
use of available training data. If T1 outperforms F0, ordinary word-aligned crop
augmentation and loss-scale controls remain necessary before attributing the
gain to a specific mechanism. Neither result alone establishes novelty or SOTA.

This phase does not modify or append tasks to the running four-model English
benchmark. Its 16 complete-cohort shards have no execution admission yet.
Prepare the code and validate the native graph on fixed short/median/long
source-development clips before considering GPU use. CPU preparation may run
only within a finite resource window and after the current T2/T4 CPU grid
checks close. No paid allocation or deadline extension is granted by this file.

Before assessment, require a frozen worker and independent verifier, complete
source calibration, a finite budget with backup, and verified completion of
the parent benchmark for the intended comparative analysis. Preserve failures
and incomplete attempts. Use the original paired clip-level uncertainty for
WildVSR, whose available release lacks an independently reconstructed source map.
