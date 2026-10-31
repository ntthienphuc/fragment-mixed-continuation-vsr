# Connect the matched schedule to the unchanged native objective

This adapter supplies the previously verified full, fixed-time, whole-word and
weighted-full inputs to the original recognizer. It is preparation for a future
training decision; it does not modify the T2/T4 matrix or admit a long run.

`adapter.py` exports `MatchedObjective(dataset, arm)` with the native-compatible
`loss(model, frames, row, sample_index, seed, budget)` signature. It requires the
unchanged T0 native graph. The schedule chooses input frames and target text;
weighted-full multiplies the native mixed objective by the exact declared ratio.
All other arms use the native objective with scalar one. It checks sample ID,
training split, seed, shape, target hash, token count and CTC feasibility before
forward. It neither changes optimizer behavior nor selects a checkpoint.

`verify_native.py` exercises actual local VLR videos and the original initializer
on CPU float32. The four probe positions are chosen from training metadata:
first full slot, first ordinary short slot, first shared fallback, and first
maximum-duration non-fallback whole-word slot. Compare the adapter against the
archived T0/T1 objective with the same model and restored random state. For the
whole-word arm, supply the independently verified slice/text as a complete
utterance to archived T0. For weighted-full, compare the explicit weighted T0
loss. Use training mode with frozen BatchNorm and reset dropout randomness;
disable gradient recording to limit local RAM. Every loss component must match
exactly. Check that model/initializer bytes remain unchanged and CUDA was unused.

Reject incorrect clip identities, validation rows and seed mismatches. Recheck
raw media hashes and canonical pixel identity before using a probe. This check
is about correctly connecting the schedule to native forward/loss computation.
It does not test backward, CUDA bfloat16, fused AdamW, exact resume, validation
integration, long-run stability or WER. The earlier CPU gradient-scaling failure
is preserved and is not overridden by a successful forward comparison.

The existing verified one-update CPU AdamW evidence is separate. A training
driver must bind this adapter and schedule to its run identity; using the old
driver unchanged would not record the new intervention. Complete the frozen
T2/T4 analysis and resolve native training/resource prerequisites before launch.
