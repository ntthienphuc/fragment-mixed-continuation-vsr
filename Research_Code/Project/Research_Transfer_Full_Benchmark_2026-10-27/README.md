# Full-clip transfer benchmark preparation

Updated 27 October 2026, 23:43 Vietnam time. This phase prepares the complete
ViCocktail benchmark while the separate seed-29 decoder matrix runs on A40.
It contains verified inputs, decoder selection, inference/task/verification code
and a complete-test analysis implementation. Expanded recognition has not started
and is not yet admitted; actual runtime preflight remains necessary.

## Why this extension matters

The completed seed-17 result concerns 531 development clips of at least 82 frames.
The next comparison must include all durations, freeze model-specific decoders on
development and score the complete official test. This distinguishes a useful
benchmark effect from a result limited to the previously examined cohort.

`Protocol.md` specifies five matched models: B0, T0/T1 seed 17 and T0/T1 seed 29,
all adaptations at step 6,000. It also includes a native ViCocktail practical
reference with its original preprocessing and graph. That target-trained model
is not a causal control for the adaptation pair. No model is chosen from the test.

## Completed checks

- All 1,935 video files match their declared SHA-256 hashes. The full population
  is 768 development clips / 43 video sources and 1,167 test clips / 20 sources.
- Every video was decoded again in both RGB and grayscale conventions: 3,870
  arrays passed frame count, shape, dtype and frame-rate checks. All 2,655 cached
  model/clip pixel hashes were reproduced, covering the five matched models on
  the prior 531 clips. This is actual video decoding, not only manifest inspection.
- All five matched checkpoint files and the native checkpoint were rehashed.
  Every one of the 767 VLR initializer tensors matches the original Lightning
  checkpoint. Its stored configuration records adaptation from the named
  Auto-AVSR LRS2/LRS3/VoxCeleb2/AVSpeech pretrained model to `benchmark_vi`.
  The native ViCocktail checkpoint records a separate training run. Metadata
  establishes declared lineage; it does not prove absence of every upstream
  pretraining overlap.
- Decoder selection was tested against 5,841 real historical predictions and six
  invalid-input cases. Tests distinguish pooled WER from mean per-clip WER,
  enforce the frozen tie order, and reject test/missing/duplicate/corrupt records.
  No full-benchmark decoder has been selected yet.

The first population-preparation attempt exposed a metadata difference: official
test label hashes live in the raw-label verification table rather than the video
manifest. `Inputs_01` preserves that failed code and diagnosis. `Inputs_02` uses
the verified label mapping and completed successfully; no labels were changed.

## Execution preparation and remaining checks

The inference adapter, admission checks, prior-prediction merger, finite queues,
test worker and paired-source analysis are implemented. `Execution_Lock.json`
freezes 213 source files and 96 complete-population tasks. Preparation tests
reconstructed 17,523 historical prediction counts, replayed 77 greedy token paths
from real caches, and separately checked 5,841 of those records through the new
independent verifier. Twenty-four invalid-input cases were rejected across the
two test suites. A real child process was stopped at its allocated deadline.

Synthetic analysis tests retain a negative second seed, distinguish pooled WER
from mean source WER, and verify the two-seed mean under identical source draws.
These test fixtures are not experimental results. The current source freeze
does not establish that the new full runtime has executed successfully.

After reviewing the complete seed-29 matrix, admit bounded runtime preflights
for the standard and native ViCocktail graphs. Measure time/memory and make a
separate resource decision before expanded recognition. The current A40 lease is
not automatically extended. See `Execution_Guide.md` for interfaces and order.

Reuse 531 development predictions only from complete verified source phases.
The remaining workload has 237 new clips for each matched model plus all 768
development clips for the native reference: 21,483 decoder outputs. On the test,
only one frozen development-selected decoder per model is needed: 7,002 outputs.
These are workload counts, not completed recognition results or measured hours.

The official test has been used elsewhere in this project. This extension is
benchmark confirmation, not a newly hidden test. Preserve both training seeds
and report source-domain tradeoffs. A positive result still needs simple
exposure/loss controls before attributing it to a new learning principle.

## Evidence and commands

Completed using the existing project ML Python environment, each with exit 0:

```text
python -B -X utf8 trace_initializer.py
python -B -X utf8 prepare.py --attempt Inputs_02
python -B -X utf8 test_selection.py
python -B -X utf8 -u verify_pixels.py
python -B -X utf8 test_execution.py
python -B -X utf8 test_protocol.py
python -B -X utf8 freeze_execution.py
```

Receipts: `Lineage_01/Summary.json`, `Inputs_02/Summary.json`,
`Selection_Verification.json`, `Pixels_01/Verification.json`,
`Execution_Tests_01/Verification.json`, and `Protocol_Verification.json`.
`Plan.json` binds input identities and the protocol. Existing output folders are
exclusive; do not rerun into them. The selection CLI requires a complete,
verified six-model development bundle and writes a single selection lock.

## CPU diagnostic completed during the server run

`State_Drift_01/Findings.md` compares the actual final states for both seeds with
the initializer. T1's floating-state adaptation norm is 1.37/1.38 times T0's,
with adaptation-vector cosine about 0.40. All running buffers and nonfloating
counters remain identical to the initializer. This rules out only the simple
stored-state explanation of T1 being a smaller move along T0's direction; it is
not a causal account of WER and does not eliminate interpolation as a recognition
control. The diagnostic used CPU, no predictions or labels, and passed independent
real-tensor checks. Its first key-prefix failure is preserved separately.

The existing interval-transcription paper and Overleaf are unchanged. No results
from an incomplete matrix are used to support a manuscript claim.
