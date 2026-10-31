# Matched continuation population analysis

These tools prepare analysis of the fixed final FC, fixed-time and whole-word
continuation checkpoints. They do not launch training, change the existing
assessment lock, or turn a resource probe into a corpus result.

## Evidence required

`collect_population.py` requires all 16 source shards or all 8 destination
shards for each requested model/readout. It independently recounts every
prediction, checks original positions, references, pixel identities, checkpoint
binding, code lock, completion hashes and fresh-replay receipts. It rejects
recovery checkpoints, incomplete coverage and mixed runtime identities.

For a selected decoder, supply all eight development-grid shards as well as the
selection file. The collector independently reproduces that selection before
reading the selected assessment result. The common decoder remains the primary
result; it is never replaced by a more favorable selected result.

`paired_statistics.py` calculates corpus WER from pooled edit counts and 10,000
paired bootstrap resamples of archived source groups. Positive differences mean
the reference model has more errors. Intervals are pointwise and conditional on
the fitted checkpoints. They are not uncertainty over a population of training
seeds or a claim of independent test data.

`summarize_population.py` writes `Results.csv`, `Per_Clip_Counts.csv`, paired
contrasts when both models exist, and a provenance-bound `Summary.json`. A
single-model output is labelled as such. Source and destination are analyzed
separately. The complete matched comparison requires all three models on both.

## Invocation

Create a JSON input specification with `assessment_plan_sha256` from the phase
plan, `stage` (`source` or `assessment`), `readout` (`common` or `selected`),
`destination_root`, and a `models` object. Each model entry contains absolute
`binding` and `shards` directories. A selected entry additionally requires
`selection` and `development_shards`. Model names are `MATCHED_FC17`,
`MATCHED_FT17`, and `MATCHED_WW17`.

```powershell
py -3.11 -X utf8 .\summarize_population.py --inputs .\Inputs.json --output ..\Analysis_01
```

The output must be a new folder under the assessment phase. Do not generate an
input specification implying a completed checkpoint before its verification and
backup have arrived. Do not substitute archived FM for the new fixed-time arm.

## Verification completed

`Regression_01/Complete.json` records the real test run:

- All locked reference metadata: VLR 11,307 / 225,960 units / 61 groups;
  development 768 / 15,919 / 43; ViCocktail assessment 1,167 / 30,091 / 20.
- Historical aggregate totals reproduced for six complete models, 1,167 clips
  each. All 10,000 paired bootstrap draws cross-checked using an independently
  constructed source-count matrix; invalid pairings rejected.
- The shard reader exercised with one real CPU-replay record inside an explicit
  synthetic transport fixture; nine malformed or inadmissible cases rejected.

The transport fixture is not recognition on a new model or corpus. It cannot
pass the production collector's final-checkpoint gate. Complete new-model
population assembly and selected-readout assembly remain to be exercised on
actual finished outputs. No manuscript number changes as a result of these tests.
