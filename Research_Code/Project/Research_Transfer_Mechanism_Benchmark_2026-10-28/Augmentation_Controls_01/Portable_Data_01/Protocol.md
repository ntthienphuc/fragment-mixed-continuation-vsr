# Portable inputs for matched continuation controls

29 October 2026. This closes the difference between the original local archives
and the verified server package. It does not change any active assessment,
training schedule, recognizer, objective, or update engine.

`portable_data.py` exposes `InputPaths(runtime_file, expected_sha256)`,
`TrainingData(objective, spec, paths)` and `ValidationData(spec, paths)`.
They implement the native sample/selection/frames interfaces. Every run must bind
the runtime receipt and provider source hashes in its own identity; no implicit
substitution of this package for a full original archive is permitted.

The training manifest remains the complete original manifest. The eligible-row
order must reproduce all 48,000 locked presentations under the native seed-17
sampler. Only the scheduled 48,000 raw videos are present on the server; requests
for other media are rejected. Pixels are decoded using the archived reader and
checked against the original pixel or frame-cache digest.

The validation archive retains the complete original 11,578-row manifest, but
contains media for exactly the first 512 clips ranked by SHA-256 of clip ID.
Keep all metadata for source-disjointness checks. Explicitly reject requests for
full validation or media outside the 512-clip subset. Native window eligibility
is applied after this ranking; report eligible and ineligible counts separately.
This subset supports the previously specified training diagnostic. It is not a
replacement for whole-corpus assessment or a new target test set.

The server-only CPU check compares the 21 real training videos with the separately
packed GPU-check frames, compares selected views/targets for all three objectives,
traverses all 512 validation caches, and exercises native threaded prefetch and
source-disjointness. It tests rejected limits, seeds, indices and paths too.
It must leave CUDA uninitialized and never instantiate a recognizer or optimizer.

The source assessment lanes and isolated CUDA resource/resume probe keep their
existing priorities. Passing this input check does not admit a 6,000-step run.
Remaining admission depends on the measured GPU check, source-retention findings,
a separately identified production driver, a finite runtime budget and backup.
