# Author completion before submission

Updated on 31 October 2026. This note accompanies the anonymous review draft;
it is not part of the scientific manuscript. The locked source and destination
short-only assessment is complete and included in the paper.

## Files and format

Use the final `Overleaf_Source.zip` with `main.tex` as its main document. The
current PDF has 35 pages. The publisher's [Pattern Recognition page](https://shop.elsevier.com/journals/pattern-recognition/0031-3203)
specifies one column, double spacing, numbered pages and 20--35 pages for an
original article. Recheck pagination after adding authors and declarations.

`Highlights.txt` contains five short findings supported by completed results.
Elsevier's [highlights guidance](https://www.elsevier.support/publishing/answer/how-do-i-include-highlights-with-my-manuscript)
specifies 3--5 bullets, at most 85 characters each including spaces, supplied as
a separate editable file. Copy the text into Word if the submission form asks
for that format. Journal-specific file requirements must be checked in Editorial
Manager: the ScienceDirect Guide for Authors returned HTTP 403 during this check.

The numerical supplement is `Data.zip` plus `Code.zip`. The portable code
reproduces exported edit-count statistics and confidence intervals. It does not
retrain or rerun neural recognition. Link the continuation study's actual release
when one is published; the older interval-transcription repository is a different
paper. The original VLR dataset record is a dataset citation, not a substitute
for the code and derived results of this study.

## Details to fill from the authors' actual circumstances

- Names in the agreed order, affiliations and corresponding-author contact.
- Each author's contributions, funding and grant details, and competing interests.
- The final publication/repository identifiers and any material permissions.
- Confirmation by all authors of the final text, findings and submission.

No claim of absent funding, absent conflicts, ethics approval or co-author
approval has been filled automatically.

## AI assistance: draft wording to review

Elsevier's [AI policy](https://www.elsevier.com/about/policies-and-standards/generative-ai-policies-for-journals)
calls for disclosure of substantive writing assistance and a Methods description
when assistance extends to the research process. This project used assistance
for planning, code and analysis as well as writing, so a language-editing-only
statement would be incomplete. The following drafts describe the work performed;
the authors should verify the tool/version information and add their actual
review and responsibility statement before submission.

Suggested Methods paragraph:

> OpenAI Codex assisted with experimental planning and with preparing and
> debugging the training, evaluation and statistical-analysis scripts.
> Recognition outputs and numerical measurements were obtained by executing
> the reported models and procedures. The research record retains the code,
> protocol settings and execution outputs used to check these measurements.
> [Authors: specify the tool/model version(s) from the retained session records
> and describe your verification of the research workflow.]

Suggested declaration before the references:

> OpenAI Codex was used to assist literature discovery, manuscript drafting,
> organization and language revision. It also assisted preparation of Python
> plotting and layout scripts. [Authors: identify the relevant versions and
> describe your review, editing and responsibility for the final article.]

For the final figure captions/Methods description, record that empirical plots
come from the exported measurements and dataset examples come from identified
video frames. State the assistance used for the workflow schematic and plotting
scripts, following the same policy. Keep the primary images and the numerical
inputs unchanged. These author statements should be integrated in the final
submission after author review; they have not been inserted as completed
declarations in the current PDF.

## Completed exploratory supplement

The fixed greedy CTC readout diagnostic is complete on all three native states
and both full populations. Include `Supplement.pdf` and `Greedy_Diagnostic.zip`
with the review materials. It is explicitly exploratory, adds no training
seed or independent test data, and does not replace the common readout.
