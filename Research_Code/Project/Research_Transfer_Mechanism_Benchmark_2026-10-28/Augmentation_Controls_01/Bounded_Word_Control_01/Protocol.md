# Conventional word-complete crops within the existing student budget

This revision resolves the extreme durations found in Readiness_02. It changes
only preparation of a future training control. The active T2/T4 evaluation,
its cohorts, decoder choices, checkpoint files and previous schedules are untouched.

## Derivation of the admissible duration

The archived dataset configs set the core to 50 frames. The native training
geometry supplies 16 context frames per side for T4. Use those already declared
values, not a percentile selected from assessment performance:

    minimum_frames = core_frames = 50
    maximum_frames = core_frames + 2 * context_frames = 82

At the existing 25 fps this is 2.00--3.28 seconds. The ordinary augmentation
control therefore has at least T1's core length and no more student frames than
T4 on a non-fallback short slot. This is an input-budget constraint, not a claim
that these durations are universally optimal or that FLOPs and label counts match.
T1 retains 50 frames; the ordinary control may receive more video. Report that
asymmetry and the actual distribution. A benefit for the ordinary control could
reflect its more complete visual/lexical evidence, not merely its boundary rule.

## Span and fallback rule

Keep the original train IDs, sampler order and 24,000 full slots per language.
On each of the 24,000 original short slots, enumerate contiguous positive-duration
word spans intersecting the original core. Do not skip a zero-duration word
inside a span. Round its first word start down and last word end up to frames,
as in Readiness_02. Require the span to remain inside the original clip and in
the derived 50--82-frame band, with nonempty native tokens and feasible CTC.

Among admissible spans, retain Readiness_02's deterministic priority: smallest
total distance from the original two core boundaries, then closest duration to
the 50-frame core, then closest midpoint, then earlier start and word indices.
Do not select among rules using recognition outcomes.

If no admissible span exists, use the full original video and its full transcript
in this control AND in every newly matched T1/T4/loss-control comparator. Never
drop the sample or silently use the archived T1 checkpoint as an exact match.
The raw source video and native tokenizer are unchanged. Automatic word timings
remain automatic labels; whole-word containment here is not human verification.

## Reuse, verification and next decision

The generator reuses Readiness_02's verified slot construction with explicit
replacement of only the span-selection function and output directory. Bind the
parent implementation hashes. The verifier reuses its record checks but replaces
the exhaustive candidate enumeration with a separate bounded implementation;
also check the duration constraint on every generated non-fallback record.

Verify all 48,000 short slots and the unchanged full-slot counts. Record new
fallback identities, frame/token exposure and how often targets differ from
original T1. If training is later admitted, propagate the shared fallbacks and
regenerate the matched loss-control weights. The original Loss_Control_01
schedule must not be mixed with this revised comparison.

Complete the frozen T2/T4 matrix before deciding which matched training controls
to run. Metadata readiness is not native backward/optimizer readiness and is
not evidence of a recognition improvement. No GPU training is admitted here.
