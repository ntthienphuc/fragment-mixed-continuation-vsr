# Draft cover letter — for author review before submission

Dear Editors of Pattern Recognition,

Please consider *Fragment-Mixed Continuation for Cross-Corpus Visual Speech
Recognition* as an original research article. The manuscript examines how the
temporal granularity of supervision during source-only continuation affects
whole-utterance recognition on both the source corpus and an unfamiliar corpus.
It preserves the recognizer architecture and inference procedure.

The study identifies a useful balance between full-utterance and short-view
supervision under a fixed continuation budget. With matched initialization,
source-clip order, optimizer updates and a common decoder, fragment mixing
improves VLR-to-ViCocktail WER by 3.41 and 3.17 percentage points across two
seeds. A separate matched comparison shows why retaining complete utterances
matters: replacing the scheduled full views with additional whole-word
fragments worsens destination and source WER by 7.328 and 6.238 points,
respectively, predominantly through additional deletions. Whole-word crop,
token-normalized loss and same-target context controls further characterize
which components support the observed transfer benefit.

These results address a practical pattern-recognition problem: adapting an
existing recognizer using available labelled source data while evaluating both
transfer and retention. The inference procedure remains unchanged. Joint
source/destination assessment quantifies the Vietnamese transfer gain and
source cost, while the LRS2-to-WildVSR comparison establishes a contrasting
regime. Measured frame/token exposure and conditional paired intervals define
the interpretation of the fixed-budget comparisons. This combination of
matched interventions, readout controls and complete-population measurements
provides an empirical basis for choosing a continuation schedule.

The anonymous manuscript is accompanied by reproducible numerical data,
scientific training and evaluation source, model and protocol identities,
and verification receipts, together with a supplementary fixed greedy-CTC
readout diagnostic. Large third-party media and model files remain
subject to their providers' access conditions.

Sincerely,

Corresponding author — to be completed by the authors.

---

Before submitting, the authors must supply their identities and contact
details, review the final text and disclosures, and state originality,
exclusive submission, author approval, funding and competing interests
according to their actual circumstances. This draft does not make those
declarations on their behalf and has not been sent to the journal.
