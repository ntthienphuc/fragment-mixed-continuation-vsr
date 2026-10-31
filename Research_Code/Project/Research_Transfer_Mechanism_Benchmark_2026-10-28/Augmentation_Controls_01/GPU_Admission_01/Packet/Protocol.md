# GPU readiness for the matched augmentation comparison

29 October 2026. This is a resource and implementation check, not a recognition
experiment and not permission to change the active source-retention jobs.

Previous goal turn: PROGRESS. It completed the independent paper review, revised
and verified the Vietnamese-focused manuscript, preserved original evidence and
created a read-back-verified backup. This turn continues live source assessment
and closes the remaining CUDA implementation uncertainty for the next control.

## Scientific decision

The main comparison remains matched fixed-time versus whole-word continuation.
Both use the already locked seed-17 training order, 6,000 updates, eight examples
per update, the same VLR initializer and the same 317 full-view fallbacks.
The current archived T1 is a contextual reference, not the exactly matched new
fixed-time arm. A full-clip arm remains a control. No weighted-full arm is admitted:
its earlier strict gradient-scaling question is unresolved.

The completed T2/T4 transfer matrix does not show a clear benefit from the extra
teacher; contextual pixels retain a benefit over FC but do not improve FM.
This supports one conventional augmentation comparison, not a new architecture
search. Source-retention outcomes are still being collected. No test outcome is
used to select the training examples, crop boundaries or this resource probe.

## Bounded execution

- Use the native archived graph, native loss and verified update engine; all
  parameters trainable, frozen BatchNorm, CUDA bfloat16, fused AdamW.
- Test the first two actual locked updates of each of full, fixed_time and
  whole_word: the same 16 source-training presentations, without validation/test.
- For whole_word, save after update 1; compare update 2 uninterrupted against a
  fresh model/optimizer restored from that checkpoint. Parameters, optimizer
  states and RNG must match exactly. Do not relax this gate on a mismatch.
- Add single-example backward checks at training-metadata-selected maximum
  full-view length, maximum native-token count, maximum whole-word span and
  first shared fallback. These probe resource extremes and are not optimizer
  training steps. Duplicate positions are evaluated once per arm.
- Record elapsed update time and peak allocated/reserved VRAM separately.
  Two early updates and a few extremes cannot certify an entire 6,000-step run;
  their timing is an initial resource estimate, not a measured learning curve.
- The original checkpoint is read-only. Probe weights are diagnostic and never
  eligible for a recognition result. No assessment data is packaged.
- Maximum wall time 30 minutes, task hard limit 32 minutes. Requires GPU memory
  use below 1,000 MiB before launch and sufficient host RAM/disk.

## Scheduling and admission

The GPU probe must wait for every English source lane to complete, every shard
backup to be verified, and the isolated six-video VLR inference probe to reach
its terminal and locally backed-up state. A failed VLR probe is inspected before
this training check is launched. Do not modify existing run identities or queues.

This folder may prepare and stage a small training-only packet while inference
continues. No new paid allocation is needed. Full training requires a separate
admission after reviewing actual probe results, source-retention findings,
available training media, validation coverage, cost and server availability.
A readiness receipt must never be described as a WER improvement or novelty.
