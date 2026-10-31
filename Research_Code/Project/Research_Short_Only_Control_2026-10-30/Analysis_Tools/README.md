# Final short-only analysis

Run only after the training backup, all 24 assessment shards and final metadata are locally hash verified.

1. `py -3.11 -X utf8 Analysis_Tools/restore_inputs.py`
2. `py -3.11 -X utf8 Analysis_Tools/analyze.py`

Run from this phase directory. Restoration refuses a partial matrix and creates a fresh Analysis_Inputs_01 directory. It verifies all training backup hashes, archive members and frozen assessment identities. The analysis recounts every new prediction against the original corpus metadata, then joins the locked existing per-clip counts. Source and destination remain separate.

The primary contrast is whole-word mixed minus short-only; the reference contrast is full-only minus short-only. Positive differences favor short-only. The original paired source-group bootstrap is reused without changing its 10,000 draws or interval convention. No equivalence threshold or new test-set designation is introduced.

`check_analysis.py` passed six checks using the OLD completed arms, including all 12,474 clip identities, independent edit recount and all six existing paired intervals. It also rejects wrong source/frame/denominator joins and incomplete restoration. These checks establish analysis plumbing, not a short-only result. The first failed check assumed top-level pixel/reference fields; the corrected check uses the frozen independent record verifier and nested encoding metadata. The failed receipt is preserved.

The inference bridge has identical Python text to its parent; its archive bytes differ only in line endings. Both hashes are recorded and checked. No runtime behavior is modified by this analysis utility.
