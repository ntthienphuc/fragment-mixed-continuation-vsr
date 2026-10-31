# Retrospective duration and output-length profile of the completed English contrast

The initializer/FC/FM full-population source comparison has already been inspected.
This is a descriptive follow-up, not a new registered test or a model-selection rule.
Use all 1,243 clips and the unchanged common `beam_c0.1` readout. Do not select a
decoder, checkpoint, training crop or a promising subset from these results.

Use duration boundaries tied to the existing recipe: below 50 frames (the
two-second fragment), 50--81 frames, and at least 82 frames (the lower native
window-training eligibility bound). All videos are evaluated in full at the
prepared 25 fps. These groups partition the cohort; they do not independently
randomize duration, source, content or difficulty. The largest-duration group
has few clips and must not be treated as equivalent evidence to the others.

For each state and group, sum S/D/I/reference units and independently count output
words from normalized hypotheses. Verify predicted units = reference units - D + I.
Report the number of clips improved, worsened or tied between FC and FM and
their exact error differences. Reconcile every stratum against the completed
whole-population summary. No post-hoc significance or cross-language causal claim.

This can locate the observed source deficit and describe transcript length.
It cannot measure CTC blank posteriors, linguistic priors, alignment quality,
or explain the Vietnamese benefit. Those require separate controlled evidence.
