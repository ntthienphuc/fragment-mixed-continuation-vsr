# Deterministic reconstruction of training exposure

The running recognition experiment remains unchanged. This CPU-only check asks
how much frame and target-label exposure differed between full-clip continuation
T0 and alternating full/crop supervision T1. Equal updates and samples are not
an equal exposure budget. The check does not train a recognizer, generate new
predictions, or claim that exposure differences cause the transfer result.

Reconstruct all 48,000 samples of each final-step run for seeds 17 and 29 from
the exact archived training manifest, deterministic sample order, crop-choice
hash and Vietnamese tokenizer. Both methods use the same
training population selected with `cohort='window'` and `train_eligible=True`.
This is not identical to the separate evaluation `window_eligible` flag. Keep
metadata for excluded rows in aggregate counts.

Record full/crop counts, input frames, target words, native target tokens and
EOS-inclusive attention positions. Report the actual training-step selection
budgets separately from final-step budgets. Compare T0 and T1 on the same sample
index and clip within each seed. Check every chosen crop and target text against
the original objective control flow with a recording model that performs no
neural computation. Such replay verifies scheduling and text selection, not
training gradients, image content or optimizer replay.

The source code uses summed CTC loss and an attention criterion normalized by
batch size, not token count. Verify the attention criterion with repeated
synthetic tokens and an independent KL calculation. This demonstrates loss
reduction semantics only. Token exposure cannot be converted directly to an
effective learning rate or optimizer update because Adam, clipping, CTC alignment
and prediction difficulty are nonlinear and differ between examples.

Use the result to design a meaningful resource/optimization control if transfer
survives the running initializer/equal-step comparison and seed replication.
Do not add another consistency objective by default: T2 already tested one.
Possible future controls must distinguish content truncation from generic
regularization and exposure, and retain source and destination recognition.

The manifest is 92,632 training clips; source receipt reports 90,519 eligible
training clips. Verify these counts and exact manifest SHA-256. No source or
destination test prediction is read, and the current paper is not modified.
