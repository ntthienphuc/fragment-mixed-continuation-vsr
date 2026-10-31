# Draft cover letter — for author review before submission

Dear Editors of Pattern Recognition,

Please consider *Fragment-Mixed Continuation for Cross-Corpus Visual Speech
Recognition* as an original research article. The manuscript examines how the
temporal granularity of supervision during source-only continuation affects
whole-utterance recognition on both the source corpus and an unfamiliar corpus.
It preserves the recognizer architecture and inference procedure.

The contribution is a controlled characterization of transfer and retention,
rather than the invention of temporal cropping. Matching initialization,
source-clip order and optimizer updates, we find repeatable Vietnamese transfer
gains from fragment mixing, accompanied by a source cost. Conventional
whole-word crops also improve transfer. Replacing scheduled full utterances
with additional whole-word fragments instead worsens both source and
destination recognition, predominantly through deletions. A paired
token-normalized control and context controls test specific alternative
explanations. The English experiment provides a contrasting outcome, with a
source cost and no clear destination gain.

These results address a practical recognition question: whether an improvement
obtained by continuing to train an existing recognizer transfers outside its
training collection, and what is lost on the source. The manuscript separates
model-state effects from decoder calibration, reports both corpus pairs and
conditional paired intervals, and explicitly limits claims to the fitted
states and reused assessment populations. It does not claim a new recognizer
architecture, an optimal mixing ratio, or universal improvement.

The anonymous manuscript is accompanied by reproducible numerical data,
scientific training and evaluation source, model and protocol identities,
and verification receipts. Large third-party media and model files remain
subject to their providers' access conditions.

Sincerely,

Corresponding author — to be completed by the authors.

---

Before submitting, the authors must supply their identities and contact
details, review the final text and disclosures, and state originality,
exclusive submission, author approval, funding and competing interests
according to their actual circumstances. This draft does not make those
declarations on their behalf and has not been sent to the journal.
