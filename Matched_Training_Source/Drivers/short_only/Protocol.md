# One final control: whole-word short-view continuation

Authorized on 30 October 2026 after the completed 33-page manuscript. This is
one explicit extension to the closed finalization scope, not a new open-ended search.

## Question and decision

Does keeping full utterances in continuation help beyond short-view supervision
alone? Compare the new nominal whole-word short-only state with the already
completed matched native full-only and whole-word mixed states, on both source
VLR and destination ViCocktail. The fixed-time mixed state is supporting context.
The endpoint is one fitted state and one complete paired assessment, irrespective
of whether its result favors mixing. Do not add a seed, sweep, loss or corpus.

## Locked training

- Same initializer, native T0 recognizer, all trainable parameters, seed 17,
  original order of 48,000 distinct VLR training clips and 6,000 updates.
- Same native CTC/attention loss, optimizer, learning-rate schedule, accumulation
  of eight, clipping, bf16, deterministic settings and final-step selection as
  the completed Linux native comparison. No target-corpus training or tuning.
- Retain every odd-slot whole-word view from the mixed schedule exactly.
- For every formerly-full even slot, compute the same deterministic 50-frame
  anchor from the eligible training starts (seed, clip ID, sample index) and nearest aligned whole-word span in
  the existing 50--82 frame band. Reuse the established candidate ordering.
- If no valid whole-word candidate exists, retain the full native view and
  record it explicitly. Never drop or replace a source video. Report the actual
  fallback fraction, frame/token exposure and full versus cropped input counts;
  this arm is nominal short-only, not a claim of zero full views.
- Check all selections and tokenizer targets before GPU start. Use a two-update
  execution boundary and verified checkpoint resume before continuing to 6,000.
  The boundary is a software/resource check, not a model-selection decision.

## Locked assessment

Use all 11,307 VLR source and 1,167 ViCocktail destination clips, with the existing
common hybrid beam-40, CTC weight 0.1, no LM, zero penalty, FP32 readout.
Reuse the frozen populations, scoring and paired source-group bootstrap policy.
Report WER and S/D/I counts, source and destination differences separately.
No decoder retuning, assessment-dependent model choice, new equivalence margin,
or claim that reused test data constitute independent confirmation.

## Resources and finish

Prepare locally before resuming Vast instance 58265651. Check current price,
balance and disk. Reserve at least USD 4; bound this extension by the smaller of
USD 3.8 and available credit above that reserve, allowing for the other stopped
instance's disk and transfer charges. Train one GPU; use both GPUs for the existing
four-worker assessment if memory checks pass. Stop after verified local backups,
or at the budget deadline; finish any remainder locally without new spending.

## Manuscript use

The question is how training-view composition changes cross-corpus recognition
and source retention without changing inference. Cropping and mixed lengths are
prior art; a controlled transfer finding can contribute without a new architecture.
Keep evidence needed to judge that question in the paper and put execution detail
in reproducibility material. Do not omit a material contrary result. If short-only
matches or exceeds mixed, describe short-view supervision rather than asserting
that mixing is necessary. No unfinished result enters the numerical tables.
