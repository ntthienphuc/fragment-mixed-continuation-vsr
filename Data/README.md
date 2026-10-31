# Numerical supplement

All recognition tables describe completed full-utterance assessments. Destination
tables retain the completed continuation results. The source tables add the
complete initializer/FC/FM assessments on 1,243 LRS2 clips and 11,307 VLR clips.

| File | Content |
| --- | --- |
| Results.csv | 30 model × decoder rows with corpus-level WER and S/D/I totals |
| Per_Clip_Errors.csv | 58,628 model/readout/clip records, 4,021 distinct clips across two settings |
| Comparisons.csv | Six FC–FM comparisons and conditional paired intervals |
| Mechanism_Comparisons.csv | Sixteen context-control comparisons |
| Mechanism_Edit_Changes.csv | Edit-count decomposition for context controls |
| Length_Counts.csv | Post-hoc length strata |
| Training_Exposure.csv | Seed-17 FC/FM/T2/T4 in VI/EN and seed-29 Vietnamese FC/FM |
| Training_Label_Coverage.csv | Automatic timestamp coverage on fragment training views |
| Model_Identities.csv | Model, checkpoint/config/tokenizer identities |
| Method_Mapping.csv | Paper names mapped to experiment identifiers |
| Dataset_Figure_Frames.csv | Real frame identifiers used in the dataset illustration |
| Source_Results.csv | Twelve model/readout rows across LRS2 and VLR, at seed 17 |
| Source_Comparisons.csv | Paired source comparisons, with separate corpus grouping |
| Source_Per_Clip_Errors.csv | 75,300 source model/readout/clip records |
| Source_Destination_Profile.csv | Same fitted states/readouts on both source/destination pairs |

Vietnamese WER counts whitespace-delimited units; English counts normalized words.
Do not pool denominators across languages. Vietnamese intervals group by 20
original-video IDs; English destination intervals resample released clips.
LRS2 source intervals resample 354 archived source identifiers; VLR source
intervals resample 61 source-video groups. These are
conditional intervals for fitted models, not confidence intervals across newly
trained seeds. Different corpora have different resampling units.

`Source_Manifest.json`, `Mechanism_Adoption.json`, `Training_Design_Provenance.json`
and the figure/identity receipts preserve the parent study's provenance. They
are not newly run measurements. `Verification.json` records the fresh numerical
recount in this revision. `Training_Exposure_Seed29_Provenance.json` covers the two
new supplement rows using previously completed deterministic exposure evidence.
Seed-17 coverage provenance is unchanged. `Source_Evidence_Verification.json`
records the original-source recount and export. Manuscript checks are in
`../QA/Manuscript_Verification.json` and `../QA/Delivery_Verification.json`.

Media and third-party checkpoints are not included or relicensed. Obtain them
from the providers under their terms. Incomplete model populations are not
included. The three-state source comparisons are distinct from the larger
English control matrix and the new matched training/assessment jobs.
Primary source readouts reuse the destination-development choices without
source-test tuning. VLR's FC/FM ranking differs between primary and common
readouts; both are retained in the tables.


## Matched crop follow-up

`Matched_Crops/` adds the completed same-runtime seed-17 full-only, fixed-time and
whole-word comparison: Results.csv, Paired_Comparisons.csv, Per_Clip_Counts.csv,
Training_Exposure.json and Model_Identities.json. Each of the three models covers
all 1,167 clips; the per-clip table has 3,501 rows. Summary.json is the immutable
analysis receipt, including its pre-export manuscript status. The current paper
adoption is recorded in ../QA/Matched_Crop_Verification.json.

The mixed arms share 317 full-clip fallback positions and have different
frame/token exposure. These are new fitted states, not aliases for the original
FM or the separate Windows whole-word model. `Matched_Source/` contains their
complete VLR source assessment: 11,307 clips, 225,960 reference units and 61
archived source groups per checkpoint. Its Per_Clip_Counts.csv has 33,921 rows.
The same folder's Source_Destination_Profile.csv joins these three states to
their own ViCocktail results using identical model bindings and common decoding.
The original root-level Source_Destination_Profile.csv is preserved separately.
Paired_Comparisons.csv contains three source contrasts with 10,000 paired
recording-group resamples. No source and destination denominators are pooled.


## Loss-reduction control

`Loss_Reduction/` contains the completed four-state native/normalized comparison
on 1,167 ViCocktail clips per model. Results.csv gives the four aggregate rows;
Contrasts.csv and Summary.json contain the five prespecified contrasts and
10,000 paired source-group resamples. Normalized_Per_Clip_Counts.csv contains
the 2,334 normalized records. Native_Per_Clip_Counts.csv preserves all 3,501
rows of the parent crop study; filter MATCHED_FC17 and MATCHED_FT17 for the
four-state interaction. The whole-word rows are not part of that interaction.

Model_Identities.json separates raw resume-checkpoint hashes from compact
inference-model hashes. Training_Exposure.json confirms that each normalized
arm has the same frame/token exposure and fallback count as its native
counterpart. Both reductions have the same fixed 6,000-update endpoint.
Intervals are conditional on fitted states and this reused assessment population;
no normalized whole-word or new source result is implied by these files.

## Completed short-only follow-up

`Short_Only/` adds the completed whole-word short-only state and retains its
three matched comparators. Each state covers all 11,307 VLR source clips
(225,960 reference units, 61 archived groups) and all 1,167 ViCocktail clips
(30,091 units, 20 groups). The common readout is unchanged. The final state was
selected at 6,000 updates; it was not selected using assessment WER.

Short-only WER is 44.844220% on VLR and 70.715496% on ViCocktail. Whole-word
mixed WER is 38.605948% and 63.387724%, respectively. Four locked contrasts
use reference-minus-short-only WER, so their negative estimates favor the
reference. Fewer substitutions and insertions under short-only are outweighed
by increased deletions. The schedule has 47,402 short views and 598 full-view
fallbacks; it does not literally remove every full utterance.

`Execution_Comparability.json` records the matched runtime and cross-host
checks. The 24 disjoint owned shards were rehashed and restored before analysis.
The separate 703-clip parity run is not pooled into either population. Bootstrap
intervals remain conditional on fitted states and reused populations, pointwise
and unadjusted. These measurements do not determine an optimal mixing ratio.

The portable primary reproduction contains 60 aggregate rows and 49 intervals;
some identities recur across comparison families. Any subsequent greedy-readout
diagnostic is kept separate from these frozen common-decoder measurements.

## License and attribution

The study's original derived numerical measurements are released under
CC BY 4.0; see `LICENSE-DATA-CC-BY-4.0.txt`. Cite the accompanying manuscript
and retain the provenance of reused inputs. This license does not cover
third-party recordings, annotations, checkpoints or upstream code.
