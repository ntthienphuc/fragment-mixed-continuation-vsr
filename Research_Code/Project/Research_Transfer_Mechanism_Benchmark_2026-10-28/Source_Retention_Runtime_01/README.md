# ZIP-backed source-test inputs

`source_loader.py` supplies the complete archived source tests for the later
final-checkpoint retention comparison: 11,307 VLR clips and 1,243 LRS2 clips.
It retains manifest order, full video length and original reference text.
It does not filter by training eligibility or tune a readout on test results.

`open_source(language)` is a context manager. Its `SourceTest.load(position)`
returns the metadata and exact uint8 frame array for one complete clip. Archives,
metadata, NPZ bytes and decoded pixels are checked against the locked preparation.
No dataset is extracted or copied. The archive handle closes when the context ends.

The verifier exercises real shortest, longest, median-length, first, last and
seeded ordinary clips; it also rejects changed NPZ/pixel identities and invalid
positions. All cohort denominators remain unchanged. These are input-provider
checks, not a new evaluation of mouth tracking, labels, models or WER.

Inference is still subject to the parent protocol: finish the active target matrix,
verify the private-host handoff, measure resource needs (including the longest
59.64-second VLR clip), and retain the frozen final models and decoder settings.
No GPU job or rental is started by this module or its verifier.
