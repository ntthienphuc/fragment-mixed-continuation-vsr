# Matched continuation with component-normalized hybrid loss

This separate driver uses the same native recognizer, update/resume engine,
initializer, seed17, ordered48000 source presentations, accumulation8 and6000
updates as the native controls. Only the per-example loss reduction changes:
0.1 CTC_sum/N +0.9 Attention_sum/(N+1). EOS is counted only for attention.
Both full and fixed_time are required in the prospective paired comparison.

The original native-production files and running private job are unchanged.
All243357028 parameters are trainable; BatchNorm statistics remain frozen.
AdamW uses learning rate2e-6, native warmup500/cosine, weight decay0.01 and
clip norm5. CUDA uses bf16/fused AdamW, disabledTF32 and deterministic algorithms.
Source diagnostics every500 updates retain the original512 selected/504 eligible
window population, and never select the endpoint: final step6000 is fixed.

Admission requires the native CUDA resource/resume proof AND a successful
normalized CUDA full/fixed-time optimizer/reference/exact-resume proof in the
same software/GPU environment. CPU checks cannot substitute for that gate.
Native and normalized checkpoints have different identities and cannot be
silently substituted or resumed into one another. CPU integration is one
normalized-full update from the original initializer; it is not a trained
recognition model. The unchanged loop retains checkpoints, logs and failures.

Equal sample order and updates do not equalize frames, tokens, clipping frequency
or effective optimization. A changed FM advantage under normalization is evidence
of sensitivity to the objective, not by itself a unique causal explanation.
Long training remains unadmitted until actual GPU checks and a measured budget
are complete. No new architecture or inference-time module is introduced.
