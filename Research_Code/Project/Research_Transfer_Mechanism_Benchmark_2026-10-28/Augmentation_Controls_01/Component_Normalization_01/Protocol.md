# Prospective component-normalization control

Native single-example reduction sums CTC loss and valid attention-token loss.
For N transcript tokens, the proposed comparison uses

    0.1 * CTC_sum / N + 0.9 * Attention_sum / (N + 1)

Attention includes EOS; CTC does not. The same rule must apply to full-clip and
fixed-time arms. Dividing the entire hybrid loss by N+1 is a different objective.
This is an objective-reduction control, not a new recognition method.

`CPU_01/Verification.json` was produced on 29 October at 14:23:57 UTC with
PyTorch 2.8.0 CPU. Five cases, including repeated CTC labels, compare values and
logit gradients against PyTorch's mean CTC reduction and the native
length-normalized label-smoothing implementation. All gradient differences are
zero in these cases; seven invalid inputs are rejected and ignored attention
positions have zero gradient.

The subsequent `Full_Graph_01/Collected_Receipt.json` records a completed CPU
check on one scheduled training video, using its 139-frame full view and its
50-frame fixed-time view. For each view, all 243,357,028 parameter gradients
were compared against a fresh native forward/backward using mean CTC and
length-normalized attention loss. Both loss values and all gradient elements
matched exactly in this check. The four remote receipts were downloaded and
verified against SHA-256 hashes. CUDA was not initialized.

Those two-view receipts did not validate optimizer updates or resume behavior.
A subsequent isolated Linux CPU check completed at 17:44:44 UTC on 29 October:
`Research_Vast_Source_2026-10-29/Linux_Execution_01/Normalized_Optimizer_01/`.
Both full and fixed-time arms used the first 16 locked source presentations,
two candidate updates, an exact step-1-to-step-2 checkpoint replay, and a fresh
two-update reference using mean CTC plus native normalized attention.
All 730,099,819 model/optimizer tensor elements per arm matched the reference
within the declared rtol=1e-5 and atol=1e-7. Complete model/optimizer/scaler/RNG
fingerprints matched exactly for the resumed candidate trajectories.
Ten receipt/code/log files were backed up with matching per-member hashes.

No normalized CUDA training has been admitted. Next: verify CUDA updates and
resume under the intended production environment, then declare the full paired
matrix and execution budget. Keep existing native production identities
unchanged. These checks are implementation evidence on the declared training
cases, not efficacy or a population-wide numerical equivalence claim.

Normalization changes relative example weighting, clipping and Adam dynamics.
It does not match frame exposure, alignment difficulty or effective optimization
steps. A reduced FM advantage under normalization would show sensitivity to the
objective, not by itself identify the unique cause of the transfer improvement.
