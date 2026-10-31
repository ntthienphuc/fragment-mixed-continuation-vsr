# Whole-utterance transfer: T2/T4 follow-up

## Current status: 29 October, 03:48 Vietnam time

The frozen T2/T4 assessment is running on the private Tailscale workstation.
Sixteen of 32 assessment shards are independently verified and backed up
locally: 14 Vietnamese and two English. All development shards and the four
decoder selections are already complete. No full T2/T4 assessment table is
available yet. Live progress and the deadline/backup state are maintained in
`../result.txt` and `../Research_Tailscale_Server_2026-10-28/`.

The completed initializer-relative reanalysis adds an important distinction:
Vietnamese T1 improves both B0 and T0, whereas English has no established T1
benefit. At the common decoder, Vietnamese B0-to-T1 gains are 6.118 and 5.972
points for the two fitted seeds; English's 0.208-point estimate has a confidence
interval that crosses zero. The source-selected English primary instead favors
B0 by 0.629 points. See [the complete comparisons and interpretation](
Completed_Initializer_Comparison_01/Findings.md). These are retrospective
comparisons of completed predictions, not additional training or a new test set.

Matched conventional whole-word and loss-weight controls are prepared, including
96,000 verified schedule positions and 16 real native CPU forward comparisons.
They have not produced trained models or recognition results. Their readiness
does not change the active T2/T4 protocol or the server deadline.

The sections below preserve the preparation history. References there to work
not yet started describe that earlier stage; they are not current job status.

## Historical preparation summary, before execution

English/F0 and the Vietnamese common-decoder comparison are complete and
backed up. Vietnamese T1 retains gains of 3.410 and 3.170 WER points under
the common decoder; English remains effectively tied with T0. No T2/T4
transfer assessment has started. The next step is actual worker/verifier
integration, workload measurement and a separately bounded allocation.

Execution revision 2 corrects the selector's calibration metadata: Vietnamese
uses destination-corpus development and English uses source-corpus development.
The decoder-selection algorithm, cohorts, candidate grid and model states are
unchanged. `Execution_Verification_02.json` checks 132 actual archived source
predictions, eight full partitions and 117 malformed cases. Revision 1 is
preserved in `Execution_Revision_02/Previous_Files.zip`; the revised lock and
its read-back backup are recorded in that directory.

This phase prepares two mechanism controls for the completed Vietnamese
T0/T1 transfer result. It remains separate from the running English benchmark
and Vietnamese fixed-decoder supplement. No new paid allocation is admitted.

## Scientific question

Mixed full/cropped training improved whole-clip transfer relative to full-clip
continuation on ViCocktail at two final training seeds. The next comparison
asks whether that benefit changes when crop training receives surrounding
visual frames (T4) or a full-view CTC self-teacher (T2). These archived controls
have the same final 6,000 updates and seed 17, but different computation.

The experiment retains both a decoder selected on development and the fixed
native decoder. It does not select whichever test column happens to be better.
The complete method and interpretation limits are in [Protocol.md](Protocol.md).

| Model training language | Calibration | Assessment | New states |
| --- | --- | --- | --- |
| Vietnamese / VLR | 768 ViCocktail development clips | 1,167 ViCocktail test clips | T2, T4 |
| English / LRS2 | 1,082 LRS2 development clips | 2,854 WildVSR clips | T2, T4 |

Keep the two calibration regimes and token units separate. Every target clip,
including short clips, remains in its original cohort. Vietnamese target data
has appeared in earlier project work; this is a follow-up on existing data.

## Preparation completed

- `Plan.json`, revision 2: all four model identities, complete cohorts,
  eleven decoder candidates and 64 shard tasks.
- `Preparation_Verification_02.json`: 299 input hashes checked, all cohort
  positions covered exactly once per model, eight malformed cases rejected,
  and the actual initial incomplete partition rejected.
- `Runtime_Validation_01/Complete.json`: four strict model loads and CPU native
  replays on a fixed short development clip per model, completed in 187.7 s.
- `Runtime_Validation_01/Backup.json`: code, plans and proof copied to C: and
  every archived file read back against SHA-256. Model weights are referenced
  by their original hashes; this small preparation ZIP does not duplicate them.
- `Source_Grid_CPU_01/Complete.json`: all four states replayed on fixed
  short, median and long source clips across eleven decoder settings;
  132 comparisons completed in 1,950.9 seconds.
- `Execution_Verification_01.json`: the record verifier accepted all 132
  archived source predictions, checked eight complete cohort partitions and
  rejected 114 malformed cases. This did not run new target inference.
- `Execution_Lock.json`: the worker, independent verifier, collector and
  decoder selector are frozen for all 64 tasks. Admission remains disabled.

The two 14-frame English probes emitted empty hypotheses in both decoder
implementations. The later source grid also covers 46- and 153-frame English
clips and 25-, 113- and 370-frame Vietnamese clips. These CPU checks do not
replace a real worker/verifier integration run or source replay on the device
that will execute the complete benchmark.

The first preparation copied a parent's incremental Vietnamese queue, covering
237 new clips while that parent reused 531 prior predictions. Such reuse does
not exist for T2/T4. `Preparation_01/Rejected_Snapshot.zip` preserves that
unexecuted plan; revision 2 covers all 768 development clips for each new state.
No parent result, active worker or target prediction was changed by this repair.

## Remaining execution work

1. Finish and verify the current English and Vietnamese anchor comparisons.
2. Run an end-to-end development-only worker/verifier integration check and
   prepare the complete-population analysis against the parent comparisons.
3. Repeat the short/median/long source replay on the intended execution device,
   checking exact input pixels, vocabulary and archived model identity.
4. Measure workload, admit a bounded allocation, and lock all four decoder
   selections before any new assessment. Preserve original attempt directories.
5. Report complete-population WER and S/D/I, including negative outcomes, with
   paired uncertainty appropriate to each corpus. Interpret alongside T0/T1
   and the stronger practical reference rather than claiming a new SOTA.

No T2/T4 transfer WER has been produced here. Passing preparation checks does
not establish a new algorithm. Generic temporal augmentation and consistency
are established techniques; the purpose of these controls is to determine what
explains the observed transfer benefit and which next intervention is justified.

## Commands actually run

From `E:/Project/Lip`:

```powershell
py -3.11 -X utf8 Research_Transfer_Mechanism_Benchmark_2026-10-28/repair_preparation.py
py -3.11 -X utf8 Research_Transfer_Mechanism_Benchmark_2026-10-28/verify_preparation.py
py -3.11 -X utf8 -u Research_Transfer_Mechanism_Benchmark_2026-10-28/run_runtime_smoke.py
```

All three exited 0. They preserve their outputs and refuse to repeat over the
same attempt. Continue with a new explicit attempt rather than deleting proofs.

The source-grid replay, `test_execution.py` and `freeze_execution.py` also
completed successfully. `Execution_Lock.json` records their prerequisite hashes.
Do not edit a locked scientific module in place; use an explicit revision if a
real integration failure requires a change.

See [Research_Interpretation.md](Research_Interpretation.md) for the current
scientific rationale and the decisions these comparisons can support.
