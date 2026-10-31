# Fixed greedy CTC readout diagnostic

Authorized after the complete short-only result, at approximately 01:16 ICT on
31 October 2026. This is an exploratory follow-up; the original common-decoder
assessment and its analysis remain frozen. It is not a new blind confirmation.

Question: does the short-only deletion increase persist when the same fitted
native FC17, WW17 and SO17 models are read with greedy CTC? No retraining, tuning,
decoder sweep, blank bias, language model, or selection using assessment results.
Use framewise argmax of the existing FP32 CTC log probabilities; collapse adjacent
identical IDs, then remove blank (0) and EOS (2001), and use the frozen tokenizer
and text normalization. EOS is not used to truncate the CTC path.

Use the same pixel-hashed full-clip VLR and ViCocktail populations as the locked
assessment. The source and destination manifests and original runtime code are
checked by the existing frozen loader/runtime. All three states must cover the
same complete population before reporting a comparison. Run each model in a
fresh process, on the same Tailscale RTX 3060, FP32, TF32 disabled, deterministic
algorithms, two intra-op CPU threads. Keep exact identities and training steps.

First run six deterministic duration-quantile clips from each population with
SO, for resource and duration measurements only. Scope admission depends only on
timing, memory, asset availability and the user's approximately 06:15 ICT report
deadline. Prefer all 1,167 destination clips for all three models; admit all
11,307 source clips only if a conservative forecast leaves at least 45 minutes
for analysis and final packaging. No convenient or outcome-selected subset.
Record admission before full-population inference. Probes are not added to full
population aggregates. Preserve incomplete attempts and do not relabel them as
complete. A failed or incomplete population is excluded in its entirety.

Report corpus S/D/I, WER, output length, blank-argmax frame proportion and mean
blank probability. These are descriptions of this readout, not proof of a
training mechanism. Compare FC-minus-SO and WW-minus-SO using 10,000 paired
archived-source-group bootstrap draws, fixed seed 20261031, percentile 95% CI,
conditional on fitted states, pointwise and not multiplicity adjusted. Include
both favorable and unfavorable outcomes. Also show greedy-minus-common WER and
edit-count changes for each identical state. Do not replace main results with
the better readout. Source groups are not verified speaker identities.

Only completed, independently recounted, locally backed-up evidence may enter
the supplementary report or paper. Neural training is not rerun by this test.
