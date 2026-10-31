# Full-corpus English training control

This phase tests an existing final checkpoint from full-sentence training on the broader LRS2 training pool.
It addresses the practical alternative that expanding ordinary full-sentence training may explain or exceed
the benefit of training with temporal crops. It does not require a new training run.

The planned comparison uses the same 1,082 LRS2 development clips and 2,854 WildVSR assessment clips as the
current English transfer benchmark. Development selects among the same 11 decoder settings; the fixed
native CTC-weight 0.1 decoder is also retained. The checkpoint is fixed at step 6,000, seed 17.
F0 is a broader-training-pool control, not a matched manipulation of crop supervision alone.

## Current status

- `Plan.json` covers all development and assessment clips in 16 tasks.
- `model_registry.json` identifies the existing F0 checkpoint, tokenizer, and training identity.
- `Core_Verification_01.json` records passed decoder/count/coverage checks without model inference.
- `run_cpu_preflight.py` checks the actual F0 state on three development clips through all 11 settings.
  It waits for sufficient RAM, has a finite timeout, and creates a verified local backup.
- The first CPU preflight completed successfully on 28 October 2026: all 33 comparisons passed,
  including the 14-, 46-, and 153-frame inputs. Strict checkpoint loading and archive readback passed.
  The receipts are in `Source_Grid_CPU_01/Complete.json` and `Source_Grid_CPU_01/Backup.json`.
- The complete 16-shard worker is frozen in `Execution_Lock.json`. The worker and independent verifier
  completed an actual CPU integration run: 33 edit-count checks and 22 fresh-pixel decoder replays passed.
  `Execution_Verification_01.json` also records 51 rejected malformed input/configuration cases.
- `Analysis_Lock.json` fixes F0-versus-T1 as the primary comparison and retains T0 and the fixed decoder
  as secondary comparisons. The paired word-error calculation passed 521 scalar reference comparisons.
  The importer was exercised on 22,832 explicitly synthetic count rows and rejected seven rehashed corrupt cases.
- The L40 source preflight also passed all 33 native comparisons. CPU and GPU returned identical tokens
  and text on these inputs. An outer wrapper rejected the equivalent device labels `cuda` and `cuda:0`;
  `reconcile_gpu_preflight.py` verified the single-device alias, rechecked 66 counts and preserved the
  original failure and outputs. The explicitly derived runtime receipt is
  `Shared_Server_Preflight_03/Reconciliation_01/Canonical_Runtime_Proof.json`.
- Full evaluation is not yet admitted. It requires a separate finite allocation; assessment also
  requires the complete parent benchmark. These checks do not establish an F0 transfer result.

Run the CPU preflight once from the project root:

```powershell
py -3.11 -B -X utf8 -u Research_English_Full_Corpus_Control_2026-10-28/run_cpu_preflight.py
```

The preflight refuses to overwrite an existing attempt. Its completion and backup receipts must both
pass before scheduling the full evaluation. Existing English and Vietnamese GPU queues remain unchanged.

## Full evaluation entry points

`run_full_task.py` executes a complete registered shard under an explicit admission. `verify_full_task.py`
recounts every output and independently repeats fixed endpoint clips. `collect_full_results.py` accepts
only the eight completed, verified shards for a stage; integration output cannot enter this collection.
`select_full_decoder.py` selects from all eleven source-development settings before assessment.

`analyze_full_f0.py` accepts the completed F0 assessment manifest, the verified parent summary path and a
new output directory. It checks the parent count table, recomputes F0 counts, and writes one combined
per-clip CSV and WER/S/D/I comparison with paired uncertainty. It refuses partial matrices. The final
comparison remains conditional on the released WildVSR clips; the available release lacks source grouping.

`Analysis_Tests_01` contains synthetic software-test fixtures, not model predictions or paper results.
No new paid pod or modification of an active experiment is authorized by the files in this folder.
