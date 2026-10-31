# Training-only annotation and supervision profile

During inspection of resource-extreme training cases, three VLR source rows
contained English sentences or repeated phrases. The paired original CSV labels
contain the same words. This motivates a bounded training-only profile before
making a mechanistic claim about fragment mixing; it does not establish that
these annotations are wrong, translated, or inconsistent with the spoken video.

Keep every active assessment, original sample and training schedule unchanged.
Read only the VLR training manifest, original train CSV, and the completed
seed-17 FC/FM presentation trace. No test predictions, target labels, new
language model, or neural inference are used to choose flags or samples.

Descriptive flags, fixed before running the counts:

- `ascii_english_cues`: all alphabetic characters are ASCII and the transcript
  contains at least three distinct words from a declared common English-function
  word list. This is a text-screening heuristic, not validated language detection.
- `repeated_fourgram`: at least three non-overlapping appearances of an exact
  four-token sequence within the transcript. Repetition can be legitimate speech.
- `majority_zero_duration_words`: at least half of nonempty supplied word rows
  have equal start and end timestamps. This is a timing property, not proof that
  the text is incorrect or a criterion for deleting it.

Report counts on all prepared train rows and on the 48,000 actually scheduled
presentations. Rebuild original FM targets with native overlap rules and native
Vietnamese tokenizer; reconcile full/FM token and frame totals with completed
exposure evidence. For each flag report both full-view and FM target-token mass.
Flagged strata overlap and cannot be summed as mutually exclusive categories.
Examples are selected deterministically by clip-ID hash, not recognition outcome.

Cross-check all prepared full texts against the source train manifest and the
three motivating examples against their individual word-label CSVs. The original
CSV text and timing can corroborate provenance, but identifying a wrong-language
annotation requires audible source or independent human verification. Cropped
mouth pixels alone are not such verification.

This analysis may motivate a matched loss/annotation control; it must not be used
to retrospectively clean assessment labels or advertise a new robust-learning
method. If another control is needed, specify how it differs from the earlier
unsuccessful partial-CTC recipe and from ordinary filtering/reweighting before
starting it. No new training is admitted by this profile.
