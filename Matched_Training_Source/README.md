# Matched continuation: source implementation

This source snapshot accompanies the paper's matched full-only, fixed-time and
whole-word follow-up, its nominal whole-word short-only control, and the paired token-normalized loss control. It provides
the actual view-selection, objective and update code for inspection and reuse.
It complements `Code.zip`, which independently reproduces the numerical tables
and confidence intervals from `Data.zip`.

## Where to read the method

| Component | File |
| --- | --- |
| Full/fixed-time/whole-word pixels and targets | `Packet/support/Native_Matched_Adapter_01/adapter.py` |
| Overlap-labelled targets, including boundary convention | `Packet/source/phase5/views.py` |
| Whole-word candidate ordering and the 50--82-frame band | `Preparation/Readiness_02/word_span.py`, `Preparation/Bounded_Word_Control_01/word_span.py` |
| Shared full-view fallback and presentation schedule | `Preparation/Matched_Schedule_01/prepare.py` |
| Native CTC/attention objective | `Packet/source/phase5/model.py` |
| Component-wise token normalization | `Drivers/normalized/objective.py`, `Drivers/normalized/reduction.py` |
| Accumulation, clipping, optimizer step and resume | `Packet/support/Matched_Training_Runner_01/engine.py` |
| Native and normalized execution settings | `Drivers/native/`, `Drivers/normalized/` |

The code is copied without mathematical or operational edits. Its original
names remain so that hashes can be compared with the executed runs.
`Training_Identities.json` identifies four native and two normalized fitted
states. `Source_Manifest.json` records each original location and SHA-256.
Entries marked `executed_packet` or `executed_driver` match recorded execution
identities. Preparation helpers are labelled `support_source_snapshot` rather
than being represented as independently replayed training evidence.

Some shared modules contain historical options that the paper did not test.
In particular, `weighted_full` is not an admitted arm in this comparison.
Original protocol comments describe the time at which the code was prepared;
the accompanying paper and numerical supplement describe completed results.

## Reproduction boundary

This is a source implementation snapshot, not a standalone neural-training
distribution. The drivers deliberately check original input manifests, schedule
hashes and resource/resume receipts. Running them on another machine requires
reconstructing the input packet and generating matching environment receipts;
the archived admission has a finite deadline and must not be bypassed.

A complete neural rerun also needs the provider-supplied Auto-AVSR/ESPnet code,
the VLR initializer and tokenizer, source videos, transcripts and word timings,
and the frozen training/validation schedules. Obtain third-party assets under
their providers' terms and use the model identities in `Data/`. Those assets,
cloud credentials and server-management scripts are not distributed here.
This snapshot alone has not been tested as a fresh neural training pipeline.
Do not interpret its successful byte verification as a new training or WER run.

For the directly runnable CPU reproduction of reported statistics, use the
separate `Code/README.md` command with `Data.zip`. That workflow needs no video,
checkpoint, GPU, cloud account or original research workspace.

Original source files in this snapshot use the accompanying MIT license.
No license to third-party recordings, checkpoints or upstream code is implied.


## Whole-word short-only control

`Drivers/short_only/` contains the five unchanged executed driver files, including
`objective.py`. `Preparation/Short_Only/` contains the exact schedule generator,
protocol and verifier associated with the checked schedule. The original odd-slot
word views are retained. Formerly-full even slots use the same word-span rule.
The arm completes 6,000 updates and 48,000 source presentations, with 47,402 short
views and 598 full-view fallbacks. It retains the native T0 loss and update engine.

`Short_Only_Training_Evidence.json` records the terminal identity, checkpoint hash
and actual exposure. This is training evidence, not an assessment result or a
claim of an independently repeated neural run. Population scores belong to the
separate numerical supplement only after its complete assessment verification.

As for the other drivers, original paths, schedule checks and finite admissions
are preserved. The short-only driver expects its verified `Schedule/` beside the
original driver folder; this inspection snapshot does not supply a new admission
or redistribute the frozen target-token schedule, videos or pretrained weights.
