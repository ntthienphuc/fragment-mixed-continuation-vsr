# Cross-Language Window Recognition — preparation protocol

Prepared 2026-10-13; server revision 2026-10-14. Datasets: VLR and LRS2.
Backbone: the existing Auto-AVSR implementation.
This is a method study; it does not change the scope or release of LipTrace/SoftwareX.

## Questions and interpretation

1. Does window supervision improve recognition on matched target windows in both languages?
2. Can posterior consistency or intermediate CTC help English despite negative VLR evidence?
3. Does training with context recover its benefit without increasing insertion/deletion errors?
4. Does a boundary-conditioned readout add value beyond matched context training?

A negative result on VLR is not a universal negative result. A positive English result alone
does not establish a language effect: corpus, labels, preprocessing and initialization differ too.
Compare each intervention against its within-dataset control, then discuss replication of direction.
Do not pool Vietnamese whitespace WER with English lexical-word WER.

## Locked method names

| ID | Name | Objective or role |
|---|---|---|
| B0 | Pretrained Reference | No optimizer updates; native model, greedy and joint decoding. |
| F0 | LRS2 Full-Corpus Continuation | English T0 architecture, all feasible train utterances; select on full-utterance WER. Separate population control. |
| T0 | Full-Clip Continuation | Full-reference ASR on each training example. |
| T1 | Window Supervision | Half full-clip anchors, half aligned target-window ASR. |
| T2 | Window Consistency | T1 plus detached full-context CTC teacher on the same physical target frames. |
| T3 | Intermediate CTC | T1 plus an auxiliary CTC head after encoder block 6. |
| T4 | Context Supervision | T1 with 16 past and optional 16 future frames; decode the target span only. |
| T5 | Boundary Readout | T4 plus a trainable temporal cross-attention readout with outside-span penalties. |
| A5 | Unrestricted Readout | Same readout as T5, with no outside-span penalties; capacity control. |

T2/T3 are language-independent variants of P23/P06, not exact reruns of tone-specific heads
or historical neighboring-window losses. T5 is a working hypothesis, not established novelty.
T4 is the main control for T5; A5 tests whether a gain comes from added temporal capacity.
All trained arms use the same initializer, selected clips, sample order, updates and accumulation
within the window-method comparison (T0–T5/A5) for a dataset/seed/stage.
F0 uses a separate English full-utterance cohort and is not a matched window-method control.
English initializes from the native LRS3-trained English Auto-AVSR checkpoint and its 5049-unit
tokenizer; Vietnamese initializes from the separate VLR checkpoint and its 2002-unit tokenizer.
Training from scratch is not required for this pretrained-model adaptation study.
Different view lengths and auxiliary passes cost different FLOPs:
log elapsed time and peak memory rather than calling the runs compute-matched.

## Training and selection

- Visual-only recognition input. Original transcripts are never replaced by ASR output.
  LRS2 main has no word timestamps; its training/validation boundaries are derived by English
  audio CTC forced alignment, preserving original words. These are automatic boundaries,
  not human-reviewed timing ground truth. Audio is not supplied to the visual recognizer.
- Separate English and Vietnamese tokenizers/checkpoints; never swap output vocabularies.
- Pilot: seed 17, 200 updates, 8 clips/update, peak learning rate 2e-6 with warmup/cosine decay.
- Main: seeds 17/29/43, 6000 updates, same accumulation, linear warmup and cosine decay.
- These are resource budgets, not estimates of convergence. Pilot timing determines server cost.
- Default trainable scope: all parameters. `heads` is a separate low-memory diagnostic track,
  never merged with the full-training table. Freeze batch-normalization running statistics.
- Strict deterministic algorithms are enabled, with verified spatial-only frontend pooling.
  The native MaxPool3D kernel with temporal size 1 is expressed as independent
  MaxPool2D calls over frames. CPU outputs and gradients match exactly, including
  tied values; GPU repeatability passed the strict probe. Checkpoint keys are unchanged.
- The differentiable CTC dynamic program executes in FP32 on CPU, with logits and gradients
  transferred to/from the GPU;
  the backbone, CTC projection and attention decoder remain on the GPU. The CUDA CTC
  backward kernel failed the strict determinism probe. Native loss normalization/dropout
  are preserved and tested; exact resume and throughput are checked again on the server.
- Tokenization uses canonical SentencePiece IDs, including the checkpoint unknown unit
  for unseen input surfaces. Original numeric transcripts are not rewritten.
- CTC/attention loss weights: 0.1/0.9. Consistency weight 0.15, temperature 2.
- Auxiliary intermediate CTC weight 0.1, divided by target token count.
  Final-frame CTC head remains the evaluation head. T2 uses the current model in evaluation mode
  with stopped gradients as teacher; this is not a separate fixed checkpoint or EMA teacher.
- Window-comparison training clips: 82..400 frames, nonempty labels, feasible CTC targets and
  at least one eligible 50-frame target span. Every exclusion is recorded.
- F0 uses all 1..400-frame English training utterances with feasible original CTC targets;
  one declared 45-frame training clip cannot be cropped by the face detector, including
  diagnostic thresholds 0.3, 0.2 and 0.1. Its ID and video checksum are fixed in the config.
  It is excluded from F0 only; it is shorter than the 82-frame window threshold.
  All 1,082 validation clips remain included. This is a crop failure, not a transcript judgment.
  it does not require word timings or a target window. Cache retention includes this cohort.
- One optimizer sample alternates full/window by sample index, identically across methods.
- Window start is deterministic from clip ID, seed and optimizer sample index.
- The pilot is engineering/development evidence, not the final paper result.
- Validate at fixed update intervals; window-method selection uses greedy window WER,
  while F0 selection uses greedy whole-utterance WER. Pilot validation may use a fixed
  hash-ranked subset (VLR 512, LRS2 all 1082); final validation uses the full split.
- Main decoder comparison is native weight 0.1. Additional fixed weights 0.3 and 0.7,
  greedy, CTC-only and length bound are separately named controls, never hidden tuning.
- Choose any decoder setting on validation only, then freeze a test plan with checkpoint hashes.
- Test evaluation requires a matching frozen plan; test transcripts are not used during training.

## Targets and latency

Primary target duration: 50 frames at 25 fps (2 seconds), shared by both datasets.
The original proposed 125-frame setting has zero eligible LRS2 main train/validation clips
with 16-frame margins; the complete train/validation length census motivated this revision.
Historical 125-frame VLR results are separate; their WER must not be subtracted from the
new 50-frame results as a method effect.
Use supplied VLR timestamps or derived LRS2 timestamps that overlap the target
interval, with half-open boundaries. Zero-duration rows are included only when their time lies
inside the interval. This is an overlap-based diagnostic convention, not corrected labels.
Report whole-utterance WER separately; window WER is not the official LRS2 utterance benchmark.
Report CER and S/D/I alongside WER; retain failed predictions in coverage accounting.

Run two separately named context budgets when resources permit: past-only and both-sides.
Both-sides uses 16 future frames, equivalent to 640 ms availability. The encoder remains
bidirectional within the available window; past-only is not proof of a stateful streaming system.
No future frames beyond the declared input span are loaded into the recognizer.

## Data and reproducibility

Use official LRS2 lists. Exclude pretrain from the first comparison because train/pretrain overlap.
Use the exact existing VLR processed archive/split, identified by hashes. Its source/signer IDs
are grouping proxies, not verified identities. The processed release is not byte-identical
to the separate Zenodo raw release; do not replace one silently with the other.
Training/validation processing does not open test media. Cross-split source/hash checks run
on metadata; reference policies and final comparison cohort are frozen before final evaluation.

Primary references match tokenizer normalization: NFC lowercase whitespace; Vietnamese
punctuation is removed with word boundaries preserved and accents retained.
English retains original punctuation, including apostrophes, before SentencePiece encoding.
Raw references remain available in prepared metadata; CER excludes spaces.
No spelling, tone, or digit repair. Record normalization because historical reports differ.
Compare source-cluster paired bootstrap intervals per dataset and seed; do not treat three
seed predictions as three independent corpora. Final reporting retains each seed and its mean.

## Sources

- https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html
- https://github.com/mpc001/auto_avsr
- https://zenodo.org/records/19705466
- Local evidence: P01–P25 mapping, assessment 2026-10-09, L05/L06 and L08/L08C.

## Readiness boundary

Local tests establish contracts; server evidence separately records strict checkpoint/tokenizer
loading, raw split coverage, derived timing coverage, cache integrity and CUDA training.
Prepared uint8 caches use lossless NPZ compression; migration checks exact pixel equality.
These checks do not establish convergence, recognition improvement or publishable novelty.
Server download and training are authorized; test scoring still requires a frozen selection plan.
