# Native runner for the matched augmentation controls

The runner preserves the native T0 recognition graph and the archived seed-17
training order. It consumes the already verified matched schedule. Available
objectives are full, fixed_time and whole_word. The weighted_full arm remains
outside this runner's admission until its separate numerical question is resolved.

The fixed long-run profile is 6,000 updates, eight presentations per update,
seed 17, AdamW with learning rate 2e-6 and weight decay 0.01, native warmup/cosine
schedule, global gradient clipping at 5, all parameters trainable, and the
archived frozen BatchNorm behavior. Source validation retains the original
window/greedy diagnostic every 500 updates; the scientific endpoint is the final
6,000-update state, not a checkpoint selected on transfer outcomes.

The explicit objective and schedule hashes belong in every training identity
and checkpoint. These are new matched runs, not replacements for the archived
T0/T1 states. No assessment data may enter the training or validation providers.
VLR is supported using its local raw training media and archived source validation
cache. English training is not enabled while its training media remain unverified.

Implementation interfaces:

- `data_io.py`: `TrainingData(objective, spec)` and `ValidationData(spec)` expose
  the native sampling/evaluation interfaces and verify each consumed payload.
- `engine.py`: `update(...)` performs the native accumulation, clipping and AdamW
  step; `save_checkpoint(...)` / `restore_checkpoint(...)` bind RNG and identity.
- `run.py`: owns immutable run identity, attempt logs, finite resource admission,
  fixed optimizer profile, source diagnostics and final completion evidence.

The initial execution is a bounded CPU FP32 integration at the first update,
using eight real training examples. It is not a GPU/bfloat16 reproduction or a
recognition experiment. Long training requires a separate measured GPU admission.
Original checkpoints, frozen schedules and active server jobs remain unchanged.
