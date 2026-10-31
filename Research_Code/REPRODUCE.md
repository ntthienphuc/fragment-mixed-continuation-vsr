# Reproduction map

All paths below are beneath `Project/`. Abbreviations in this guide refer to
the actual dated directories, not additional software packages.

| Paper component | Scientific entry points |
| --- | --- |
| Original source preparation, tokenizer and eligible views | `Research_Phase_05_2026-10-13/phase5/prepare.py`, `importers.py`, `views.py`, `text.py` |
| Original English word timings | `Research_Phase_05_2026-10-13/server_2026-10-14/align_lrs2.py`, `align_engine.py` |
| Original FC/FM/FM-ST/CP training | `Research_Phase_05_2026-10-13/phase5/training.py`, `objectives.py`, `model.py`, `runtime.py` |
| Vietnamese full-population destination assessment and decoder selection | `Research_Transfer_Full_Benchmark_2026-10-27/run_task.py`, `select_decoder.py`, `analyze_assessment.py` |
| Vietnamese common readout | `Research_Transfer_Decoder_Anchor_2026-10-28/Analysis_01/` and frozen anchor plan |
| English destination assessment | `Research_English_Transfer_Benchmark_2026-10-28/run_task.py`, `select_decoder.py`, `analyze_complete.py` |
| English full-pool control | `Research_English_Full_Corpus_Control_2026-10-28/` |
| Context controls and paired analysis | `Research_Transfer_Mechanism_Benchmark_2026-10-28/run_full_task.py`, `Analysis_01/` |
| Complete source retention | `Research_Source_Retention_2026-10-29/Assessment_01/`, `Analysis_01/` |
| Matched crop preparation and native loss | `Research_Transfer_Mechanism_Benchmark_2026-10-28/Augmentation_Controls_01/Matched_Schedule_01/`, `Bounded_Word_Control_01/`, `Native_Matched_Adapter_01/` |
| Matched native training | `Research_Vast_Source_2026-10-29/Linux_Execution_01/Driver/` |
| Component-wise token-normalized training | `Research_Vast_Source_2026-10-29/Linux_Execution_01/Normalized_Driver_02/` |
| Short-only schedule, loss and training | `Research_Short_Only_Control_2026-10-30/prepare_schedule.py`, `verify_schedule.py`, `Driver/` |
| Matched/normalized assessment and group bootstrap | `Research_Matched_Continuation_Assessment_2026-10-30/`, `Research_Normalized_Continuation_Assessment_2026-10-30/` |
| Short-only assessment and comparisons | `Research_Short_Only_Control_2026-10-30/Assessment/`, `Analysis_Tools/` |
| Exposure and annotation coverage | `Research_Transfer_Training_Exposure_2026-10-27/`, `Research_English_Training_Scope_2026-10-28/`, `Research_Training_Label_Profile_2026-10-29/` |
| Published count-based statistics | `PatternRecognition_CrossCorpus_Draft_2026-10-29/Code/reproduce.py` |
| Paper tables, figures and PDF | `PatternRecognition_CrossCorpus_Draft_2026-10-29/Tools/` |

## Original training cases

Run module commands from `Research_Phase_05_2026-10-13/`. The source maps FC to
T0, FM to T1, FM-ST to T2 and CP to T4. The matched English full-pool baseline
also uses T0, with a full rather than fragment-eligible source cohort. Original
VI FC/FM use seeds 17 and 29; the remaining original reported training uses 17.
Every reported continuation endpoint has 6,000 optimizer updates and gradient
accumulation of eight presentations. Preserve the original initializer,
prepared manifest, source order, native objective and frozen batch statistics.

The training CLI is directly inspectable without launching training:

```text
python -m phase5.training --help
```

Use the archived dataset configs and prepared manifests, followed by the
preflight and preparation entry points. The initial VLR state has SHA-256
`b38177ae98f1f3cbea1a4c183c2bca804cf0a576e99abd23aab85ae56f413280`;
the English initializer is identified separately in its config. Supply source
data and timing annotations under their provider terms. The frozen configs
resolve paths relative to this phase directory. Changing a config or preparation
changes its identity and requires its own verified output directory.

An example invocation shape for a new, explicitly identified rerun is:

```text
python -m phase5.training --config configs/vlr.json --method T1 --seed 17 --steps 6000 --accumulation 8 --lr 0.000002 --eval-every 500 --validation-limit 512 --cohort window --selection-view window --scope all --precision bf16 --fused-adam --device cuda --workers 2 --threads 2 --output runs/new_vlr_T1_17
```

This illustrates argument wiring; it is not a claim that a newly configured
environment or data copy reproduces an archived run bit for bit. Inspect the
actual saved identity and compare it with the archived state. Report the final
endpoint, not a test-selected or arbitrarily favorable intermediate checkpoint.
Use fresh directories and retain interrupted attempts.

## Matched native, normalized and short-only cases

The canonical packet lives under
`Research_Transfer_Mechanism_Benchmark_2026-10-28/Augmentation_Controls_01/GPU_Admission_01/Packet/`.
It preserves the same native model code plus schedule adapters and update engine.
Some packet pixel fixtures and full training archives are external media assets;
the packet manifest checks their bytes. The short-only schedule is included under
`Research_Short_Only_Control_2026-10-30/Schedule/`. There are 598 fallbacks,
including 317 shared with the mixed schedule; no clip is removed from the order.

`Portable_Data_01/portable_data.py` opens the staged source archives.
`Portable_Data_01/verify_runtime.py` verifies real data and generates an input
proof before a driver is invoked. Each `Driver/run.py` accepts `--mode inspect`,
`integration` and `train`, together with `--packet`, `--provider`, `--runtime`,
`--spec`, `--input-proof`, `--checkpoint`, `--output` and `--arm`.
For the native driver the arms are `full`, `fixed_time`, `whole_word`; for the
short-only driver use `short_only`. The normalized driver is a separate source
and identity, rather than a command-line switch that mutates the native run.

First validate the actual data/runtime, then use inspect mode to check identities
without an optimizer update. The integration mode performs the explicitly bounded
CPU update supported by that driver. A GPU run additionally requires `--admission`;
the JSON contract is implemented by `configuration.admit`. Use a fresh deadline,
verified identity and resource measurements for a new rerun. Preserve all schedule,
initializer, source-order and endpoint checks. The original admissions have expired.

## Inference and statistics

The locked common readout is the original beam decoder at CTC weight 0.1,
beam width 40, zero length penalty and no external language model. The original
model-specific selected readouts are separate calibrated systems. Matched and
short-only comparisons did not repeat decoder selection. FP32, TF32 disabled,
the same transforms and hash-matched full-clip pixels are required for comparing
fitted states. The inference code refuses mismatched assets and graph families.

Supply each frozen registry's required model and data paths. Bind a model only
after checking its final checkpoint identity. Run probes and full tasks into
fresh directories, use the corresponding task verifier, then aggregate only
complete populations. Preserve 11,307 VLR and 1,167 ViCocktail clips for the
matched comparison. The duplicate cross-host parity run is never an extra
population shard. Source and destination WER denominators remain separate.

For the published statistics, `verify_release.py --data ...` is the recommended
portable entry point. It independently recounts the distributed per-clip edit
counts and reproduces the paired intervals. Reference/hypothesis recognition
must be rerun with the neural assets to test inference again; numerical
reproduction alone does not do that. Intervals condition on fitted states and
are not training-seed uncertainty or multiplicity-adjusted confirmation.

The paper's `Tools/` contains both executed authoring scripts and historical
revision helpers. Use the supplied final source ZIP to rebuild the PDF. Generate
new figures/tables into a new directory with the corresponding script's CLI;
do not replay historical text-rewrite helpers over the final manuscript.
