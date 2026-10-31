# Fixed greedy CTC readout diagnostic

Anonymous exploratory supplement to *Fragment-Mixed Continuation for
Cross-Corpus Visual Speech Recognition*. The main common-decoder results remain
primary. This diagnostic uses the same three fitted native FC/WW/SO states and
the complete VLR and ViCocktail populations. It adds no training seed or
independent test collection and did not select among alternative readouts.

Read `Supplement.pdf` for the protocol, six aggregate rows, four conditional
paired intervals, edit counts, output lengths and blank statistics. Full
precision results and all recounted clip-level measurements are in `Results/`.
Every model covers 11,307 source clips and 1,167 destination clips. Resource
probes are not additional observations.

## Portable reproduction

Use Python 3.11+ and NumPy. The tested numerical environment is Python 3.11.9
and NumPy 2.0.2. From this extracted directory:

```text
python Code/verify_supplement.py --package . --output Recomputed_01
```

Use a fresh output directory. This hashes the package and independently
reconstructs all 37,422 saved clip/model paths, transcripts and edit counts,
checks every population and runtime identity, recomputes the four 10,000-draw
group-bootstrap intervals, and compares the numerical exports byte for byte.
It needs no GPU, neural checkpoint, video archive, research workspace or network.
Saved blank-probability sums are range-checked, not recreated from full logits;
the logits were not archived. Blank-argmax counts are recreated exactly.

The component command for one complete corpus is:

```text
python Code/analyze.py --inputs Data/Inputs --collected Data/Collected --stage source --output New_Source_Recount
```

Replace `source` with `assessment` to recount ViCocktail. Neither command accepts
an incomplete three-model population. The inference code, original locked
protocol and bindings are under `Project/Research_Greedy_Readout_Diagnostic_2026-10-31/`.
To inspect the complete neural dependency tree, place that directory beside the
other phases in the main `Research_Code.zip` package's `Project/`. The saved
admission is historical; a new inference run requires its own measured,
explicitly bounded admission and the original hash-matched media/checkpoints.
Do not change the scientific checks to make an unverified rerun pass.

`Code/prepare_analysis_inputs.py` and `Code/finish_analysis.py` record how the
original local metadata and backups were assembled. `Code/make_supplement.py`
and `Code/package_supplement.py` preserve the document/table generation and
package-verification source, with the original template. These are provenance
tools for the original phase layout, not the portable recount entry point above.

## Building the document

Compile `Source/main.tex` with pdfLaTeX. The source contains no external media,
Python dependency, shell escape or absolute-path requirement. Two normal
pdfLaTeX passes resolve the document. `Source/results.tex` contains the table
values checked against the full-precision exports.

## Provenance and scope

The protocol and resource-based admission were fixed before population
inference. Each model/corpus used a fresh process on the same Tailscale RTX 3060,
FP32 and TF32 disabled. The locked loader checked the original pixel identity;
the independent analysis checks those identities against the frozen cohort.
All six terminal populations were backed up locally before final inclusion.
Historical terminal and backup receipts retain their status at collection time,
including the then-pending independent verification. `Results/Summary.json` and
the final delivery verification identify the subsequent completed recount;
the earlier receipts have not been rewritten to erase that sequence.

This is an exploratory readout check after the short-only result. Its intervals
condition on fitted states and are pointwise, without multiplicity correction.
Archived recording groups are not asserted to be verified speakers. The
document must not be read as a new blind confirmation or proof of one internal
training mechanism. Dataset and model access remains governed by the original
providers; this package contains derived measurements, references and predictions,
not the video corpora or large model weights.
