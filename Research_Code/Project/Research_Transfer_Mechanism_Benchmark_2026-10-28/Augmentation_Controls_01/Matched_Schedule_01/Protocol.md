# One shared schedule for the next augmentation comparison

This preparation applies the already specified bounded-word fallback to every
future comparator. It does not revise any archived training run or the active
T2/T4 evaluation. No assessment prediction is read and no training is launched.

Use the original seed-17 order: 48,000 presentations, eight per optimizer update,
6,000 updates. The original 24,000 full slots are unchanged. On each odd slot,
reuse the verified Bounded_Word_Control_01 selection. If it has no admissible
complete-word span, every arm uses the original full video and full target.
Otherwise, the fixed-time arm keeps its original 50-frame input and overlap
target; the whole-word arm uses its frozen 50--82-frame span and target.

The four descriptors are:

| Descriptor | Input and target | Scalar on native mixed CTC/attention loss |
| --- | --- | --- |
| full | Original complete clip and transcript | 1 |
| fixed_time | Original full slots, fixed-time crops, shared full fallbacks | 1 |
| whole_word | Original full slots, bounded whole-word crops, shared full fallbacks | 1 |
| weighted_full | The same input and target as full | (U_fixed_time + 1)/(U_full + 1) |

U is the native token count; +1 accounts for the attention EOS prediction.
Store integer numerators and denominators without clipping or decimal rounding.
The ratio is exactly one on full and fallback slots. This matches a hypothetical
unit loss per attention position; it does not equate CTC losses, gradients,
Adam updates, compute or learning rates. Frames and tokens remain separate totals.

Each compressed JSONL record binds clip ID, position, training split, seed,
intervals, target hashes, token counts and scalar. Weighted-full inputs are
defined by the full descriptor rather than by a second copy of those fields.
The manifests and native tokenizers remain the source of text and token IDs.

The full descriptor must reproduce T0 exposure exactly. Fixed-time is a new
matched schedule: archived T1 does not contain these shared fallbacks and is
therefore a reference, not an exact comparator. A rerun of unchanged T0 is not
automatically necessary. Decisions about a new training matrix wait for the
complete frozen T2/T4 assessment and native runtime/resource readiness.

The independent verifier reconstructs targets from the original training
manifest and native tokenizer, checks all 96,000 presentation positions, and
recounts rational weights, CTC feasibility and arm exposure. The verified
bounded selection is reused; selecting word spans is not tuned again here.
This establishes schedule correctness, not video loading, optimizer readiness,
recognition benefit or a new augmentation principle.
