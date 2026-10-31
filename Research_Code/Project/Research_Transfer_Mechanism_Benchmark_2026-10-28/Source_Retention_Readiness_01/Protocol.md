# Source-retention preparation for the final-state transfer study

This phase prepares a missing comparison while the locked T2/T4 target assessment
continues. It does not admit inference, change the target table, or train a model.

The question is whether a state that transfers better retains recognition accuracy
on its original source corpus. Use the exact final-step states and frozen readouts
from the transfer comparisons. Historical best-checkpoint greedy scores cannot
answer that matched question.

## Populations and readouts

Prepare all 11,307 VLR test clips in the preserved processed workspace split and
all 1,243 official-list LRS2 test clips in the existing cache. Include full clips
regardless of window/CTC-training eligibility. Do not shorten the source test or
select examples using recognition outcomes. This is whole-utterance recognition.

Bind VLR B0, T0/T1 final states at seeds 17 and 29, and T2/T4 final states at seed 17.
Bind English B0, T0/T1/T2/T4/F0 final seed-17 states and the stronger B1 reference.
These are separate within-language comparisons. Retain both existing readouts:
the decoder already selected for transfer and the common `beam_c0.1` decoder.
Deduplicate an identical decoder request for a model. Do not select a source-test
decoder or checkpoint after looking at the new results.

The Vietnamese transfer decoder was selected on target-corpus development; the
English decoder was selected on source development. This preparation preserves
that distinction. Source retention under a transfer-selected decoder is not the
best achievable source score after retuning. Do not mix its interpretation with
the historical best-checkpoint greedy experiment.

## Readiness evidence

Rehash both archives against their preserved acquisition receipts. Verify every
archived payload against its manifest, read every cached NPZ, and check uint8
frame shapes and exact frame-file identities. Check full cohort coverage, unique
IDs, reference-unit denominators and train/validation ID/source/hash overlap.
Record that source IDs are provided grouping labels, not independent proof of
human identity or recording independence. Preserve the VLR directory-group
correction. Reading cached frames does not independently redo mouth tracking.

Rehash model states, tokenizers and configs against the frozen registries; copy
their identities and selected decoders into a preparation plan. Do not claim model
deserialization, cross-device parity, or successful recognition from file hashes.

## Later execution and reporting

Inference may be admitted only after the active target matrix, backup handoff and
a measured resource budget. A future runner must use the same native FP32 encoder,
beam 40, no external language model, penalty 0 and full-encoded-length horizon.
Save all predictions and S/D/I counts; independently replay fixed clips and recount
the complete populations. Report source and target effects together. Additional
source-test results cannot turn a negative target result into a transfer gain.

The source tests have already been used in this project. This is a retrospective
final-state comparison on preserved splits, not a new blind test or independent
external replication. No claim of verified unseen VLR speakers is supported by
directory-derived labels. No new human transcript/timing review is supplied here.

Stop preparation on a hash, coverage or frame mismatch and retain its failure.
No GPU or paid allocation is part of this phase. The expected local workload is
archive streaming and checkpoint hashing; avoid extracting duplicate corpora.
