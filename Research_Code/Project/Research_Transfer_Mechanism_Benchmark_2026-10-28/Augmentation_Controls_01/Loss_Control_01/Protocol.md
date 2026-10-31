# Full-video loss-scale control

This is preparation of a conventional control, not a new VSR method or a training
admission. The running T2/T4 study and all earlier attempts remain unchanged.
No incomplete assessment predictions are read for this preparation.

## Question

T1 replaces half of T0's full-video presentations with two-second crops. The
native attention criterion sums losses over non-padding target positions and
divides by batch size, rather than target length. Short targets therefore change
the scale of that term in addition to changing the visual evidence and labels.
Can a full-video control with the corresponding deterministic scalar weighting
account for some of the transfer benefit?

## Derived intervention

Let U_full and U_crop be the native tokenizer lengths of the original full and
T1 target strings. Attention predicts the target plus EOS, so the position counts
are U_full + 1 and U_crop + 1. Define the per-presentation scalar as

    w = 1                                      on original full slots
    w = (U_crop + 1) / (U_full + 1)              on original crop slots

The proposed control always uses the full original video and full target:

    L_weighted_full = w * (0.1 * L_CTC_full + 0.9 * L_attention_full).

The 0.1/0.9 mixture comes from the archived training objective. The formula for w
comes from the number of attention predictions, including EOS; it is not a fitted
coefficient. Store the exact integer numerator and denominator, do not clip the
ratio or round it for training. If a crop has more tokens than the full string
under separate SentencePiece tokenization, retain and report the ratio above one.

Under the deliberately simplified assumption of equal loss per attention
position, the weighted full attention sum has the same magnitude as the short
attention sum. This does NOT equate CTC path losses, input frames, optimizer
updates, gradient direction/norm, Adam moments, clipping, dropout or BatchNorm.
It is not an assertion that the actual effective learning rate equals w.

Retain the same original IDs, order, initializer, optimizer, steps and learning
rate schedule as T0. Preserve original full targets even when the odd slot has
a short scalar. Full-video computation remains T0's computation. At inference,
read the complete video with the same frozen development/assessment procedure.

## Preparation and decision

Generate 48,000 scalar records for each seed-17 training schedule (VLR and LRS2).
Use only original train manifests, archived sample traces and the native
tokenizers. Reconcile all full/crop token totals with the earlier independently
verified exposure reports. An independent verifier must check every position,
exact rational arithmetic, split/source identities and input hashes.

This control uses the original T1 schedule, including its original odd-slot
targets, not the two fallback replacements in word-aligned Readiness_02. A later
joint augmentation comparison must explicitly propagate any shared fallback to
all its arms and regenerate the relevant scalar schedule. Never silently mix
the original schedule and a revised complete-word schedule.

Do not enqueue training merely because the scalar file exists. First interpret
the complete T2/T4 matrix, finalize the conventional augmentation duration and
fallback rule, and budget the resulting matched controls. A real native
loss/backward and optimizer compatibility check must precede a training run.
This metadata preparation alone does not establish that compatibility or WER.

If weighted full training accounts for T1, the interpretation should emphasize
optimization/supervision scale rather than unique partial-evidence learning.
If it does not, ordinary temporal augmentation remains a separate explanation.

