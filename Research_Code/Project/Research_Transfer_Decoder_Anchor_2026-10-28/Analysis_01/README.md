# Vietnamese common-decoder consolidation

The analysis compares all six models on the complete 1,167-clip ViCocktail assessment population,
using the fixed `beam_c0.1` decoder. Forty new verified shards and eight native-reference reused shards
are required. It refuses a partial matrix before parsing target prediction counts.

The selected-decoder primary result remains unchanged. This secondary table uses the same 20 video
source groups, 10,000 bootstrap draws and seed 20261027. The two trained seeds share source resampling;
their average is conditional on those two fitted states, not an estimate over all possible training seeds.

`Analysis_Lock.json` identifies the canonical code and `Verification_03.json` tests. The test checked
7,002 explicitly synthetic count rows, 202 scalar bootstrap comparisons, seven malformed cases, and
the actual missing-task gate. Closed real prediction files were read as bytes to check their hashes;
the test did not parse their error counts. This clarifies the wording of the retained first test receipt.

`wait_finalize.py` waits for up to six hours, performs the complete analysis once, backs up and reads
back its evidence, then updates `result.txt`. It does not launch recognition, change any decoder,
or revise the manuscript. Its current state is in `Watcher_Status.json`.

Final outputs, when complete, will be `Results_01/Summary.json`, `Results_01/Per_Clip_Counts.csv`,
`Results_01/Task_Manifest.json` and the phase-level `Completion_Receipt.json`. These outputs do not
exist merely because the preparation tests passed. Backups go to
`C:/Users/thien/Lip_Research_Backups/VI_Anchor_20261028`.
