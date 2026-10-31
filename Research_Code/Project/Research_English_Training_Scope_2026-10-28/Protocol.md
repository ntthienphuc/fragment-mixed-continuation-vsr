# Source-cohort and exposure reconstruction

This read-only analysis reconstructs the training population and sample schedule
of the existing English T0/T1 checkpoints at 6,000 updates, seed 17, accumulation
8. It does not train, decode, or change the running English benchmark.

Use the original LRS2 train manifest and receipt in Source_Results_Metadata.zip.
Verify its content hash against both checkpoint identities. Reuse the original
PreparedData.sample and sample_loss control flow with frame placeholders and a
recording recognizer. Independently reconstruct every shuffled epoch, crop index,
word-overlap target, and tokenizer count. No image, model layer, gradient, or
optimizer is executed. This cannot reproduce training loss or optimization.

Report prepared, full-eligible and window-eligible clips; exclusion reasons may
overlap and do not imply annotation errors. Report actual unique clips, repeats,
frames, labels and partial-word boundaries under the stored automatic timings.
Those boundaries are not human-verified articulation boundaries.

All outputs go to a new attempt directory. A separate verifier must reread the
archive and replay the trace without importing the reconstruction implementation.
The findings explain the scope of a matched comparison; they must not change
the already frozen benchmark populations, decoder choices or success criteria.
