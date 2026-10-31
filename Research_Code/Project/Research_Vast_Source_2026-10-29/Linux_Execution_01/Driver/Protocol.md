# Production driver for the matched augmentation comparison

This driver connects the verified portable source inputs to the unchanged native
recognizer and update engine. Previous goal turn: PROGRESS, completing the
training-only label profile, the follow-up review and verified local backups.
No data filter, new loss, model component or decoder setting is introduced here.

The scientific arms are full, fixed_time and whole_word, using seed 17, the same
48,000 ordered source presentations, 6,000 optimizer updates and accumulation 8.
Fixed-time and whole-word share the verified 317 full-view fallbacks. The archived
FM state remains a contextual reference. Final update 6,000 is the recognition
state; source diagnostic WER does not choose an earlier checkpoint.

Native settings remain all trainable parameters, frozen BatchNorm statistics,
AdamW, learning rate 2e-6 with native 500-step warmup/cosine schedule, weight
decay 0.01 and gradient clipping 5. CUDA training uses bf16 and fused AdamW,
deterministic algorithms and disabled TF32, matching the queued resource probe.

`configuration.py` validates packet, provider, source inputs, disjointness and
the fixed diagnostic cohort, then builds the complete run identity. `loop.py`
uses the existing `engine.update`, checkpoint and RNG functions. `run.py`
supports inspection, one CPU integration update, or explicitly admitted CUDA
training. Production mode requires an identity-matched admission with an actual
successful CUDA resource/resume receipt and a finite deadline; this file does
not supply that admission. It cannot import CPU diagnostic weights into training.

Source diagnostic validation is scheduled every 500 steps with the unchanged
native greedy/window/both-context procedure. Select the first 512 source-val
clips by the original ranking, then score the 504 eligible windows. Check all
identities, reference units, zero inference failures, 512 requested rows and
8 ineligible rows. Preserve output predictions alongside the metric summary.
This is a diagnostic subset, not full validation or a held-out transfer test.

Persist a checkpoint at the declared stopping step and validation boundaries.
Resume only the same identity and environment from its last complete checkpoint;
preserve earlier attempt logs and never overwrite a completed run or an
interrupted checkpoint temporary file. A finite stop produces an incomplete
receipt, not a full training result. Runtime/error receipts remain separate.

CPU integration is exactly one whole_word FP32/unfused update from the original
initializer, with CUDA hidden before imports. It is a driver check; its weights
cannot enter recognition tables. The already queued GPU probe remains unchanged.
After measured GPU admission, a real bounded production start still needs
monitoring for cost, output coverage and checkpoint/backups before proceeding
through the full endpoint. No paid cloud is allocated by these scripts.
