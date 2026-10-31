# Source retention and transfer, evaluated together

This retrospective analysis uses the already frozen final checkpoints and
decoder settings. Its question is whether the full-utterance transfer behavior
is accompanied by a gain or loss on the original source corpus. It does not
choose a decoder, checkpoint, training schedule, or subset from source-test WER.

The source cohorts are the complete 1,243 LRS2 and 11,307 VLR test clips. A model
is summarized only when every assigned shard has a completion receipt, an
independent edit-count check and fresh-pixel replay, and a verified local backup.
Incomplete shards never contribute a population WER. One completed model may
be inspected while other models run, but that is not a completed comparison.

Report the frozen primary decoder and the common CTC-weight-0.1 decoder
separately. Primary settings can differ by model; the common setting is the
cleaner comparison of fitted states under an identical readout. Source and
destination WER have different denominators and must not be averaged together.
Use the same fitted state and decoder when relating the two domains.

Main contrast: FC minus FM, with a positive difference favoring FM. Also retain
initialization versus FC/FM, FC versus FM-ST/CP, and FM versus FM-ST/CP. The
English full-pool and larger-data references have different training resources
and are descriptive controls. The Vietnamese second seed remains separate.

Confidence intervals use 10,000 paired bootstrap samples of source groups:
354 archived LRS2 source identifiers and 61 archived VLR source identifiers.
These labels are not asserted to be independently verified speaker identities.
Each sampled group retains all its clips; WER is recomputed from summed errors
and reference units. Report pointwise 95% intervals conditional on each fitted
state, without an across-seed or multiplicity-adjusted significance claim.

The present tests have been used previously. This protocol is a retrospective
analysis specification, not a registration or a fresh blind evaluation. The
source grouping differs from the existing WildVSR clip-bootstrap analysis;
interval widths across domains are therefore not directly comparable.

Here, source means the corpus used for continuation: LRS2 for English and VLR
for Vietnamese. The English initializer itself was trained on LRS3. LRS2 results
therefore do not measure forgetting on the initializer's upstream LRS3 training
domain. The current manuscript already makes this distinction; keep it explicit
when interpreting the new source/destination table.

## What another training control could still answer

The completed transfer table supports FM over matched full-clip continuation
in Vietnamese, but not a reliable English aggregate gain. FM-ST did not add
benefit. CP recovered nearly all training-label coverage yet was worse than FM
in Vietnamese; CP also changes visual context, so it does not isolate boundary
label truncation by itself.

The prepared whole-word control is still informative: compare whole-word crops
with fixed-time crops using the same archived presentation order, optimizer,
6,000-step endpoint and shared full-clip fallback positions. The archived T1
run lacks those shared fallbacks, so it is a reference rather than the exact
comparator. Whole-word spans contain 50--82 frames rather than exactly 50:
duration, context and label completeness remain coupled. A positive result
would establish an augmentation comparison, not a unique causal mechanism or
a new principle of word-aligned cropping.

No new long training is admitted by this note. First finish source retention,
then use an isolated GPU resource/gradient/resume check for any matched rerun.
The weighted-full gradient issue remains unresolved and is not waived by the
successful CPU forward or one-update checks of the other arms. English training
also requires verified training media. Avoid adding another teacher or context
module before this simpler comparison and its scientific purpose are settled.
