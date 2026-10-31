# Matched continuation: full-utterance source and destination assessment

The question is whether the particular fixed-time supervision design improves
transfer beyond conventional whole-word fragment mixing. The old FM destination
gain motivates this test; it does not answer it. Do not add an architecture or
change the training schedule while these controls are pending.

## Comparisons

Use the completed 6,000-step endpoints of full-clip continuation (FC), fixed-time
fragment mixing (FT), and whole-word fragment mixing (WW). The new three-arm
schedule shares 48,000 source presentations and 317 fallback positions. It does
not equalize frame exposure, token exposure, loss magnitude or GPU time. Record
those quantities and the training runtime. The archived FM state used a different
fallback schedule and remains a historical reference, not the new FT arm.

The primary comparison uses the same frozen hybrid decoder, beam_c0.1, for every
model on both source and destination. Beam size is 40, length penalty zero, no
external LM, float32 inference, and the search horizon is the whole encoded clip.
This setting matches the existing common-decoder evidence and is fixed before
scoring the new endpoints. Always retain this comparison even if an individually
selected decoder produces a different ranking.

The secondary comparison gives every new model the existing eleven-candidate
grid on all 768 destination-development clips. Select by total errors; resolve
ties in the frozen candidate order. Freeze that selection before assessing the
1,167 destination clips and use it unchanged on source. Do not borrow an old
model's best decoder, select on either assessment set, or choose whichever
readout happens to favor the proposed method. This grid remains required for
the separately calibrated result; a common-decoder result alone is not that result.

## Populations and evidence

Source is the same 11,307 VLR clips, 225,960 reference units and 61 archived source
groups used in the ongoing initializer/FC/FM matrix. Destination is the same
1,167 ViCocktail clips, 30,091 units and 20 groups. Original manifests, pixels,
references and group definitions are reused unchanged. These are repeatedly used
assessment sets, not a new blind confirmation.

Report each corpus separately. Primary contrasts are WW minus FT, FC minus FT
and FC minus WW; positive differences mean the second named model has fewer
errors. Independently recount all predictions, substitutions, deletions and
insertions, then compute paired source-group bootstrap intervals with 10,000
resamples. Require complete population coverage and deterministic replay before
promoting a table. Do not treat nonsignificance as equivalence or infer a unique
boundary mechanism from an augmentation comparison.

## Runtime and execution

The new training identity uses native_graph=T0 rather than the older method
field. runtime_bridge.py binds that field explicitly without editing the training
identity. Encoding, CTC, beam search, normalization and vocabulary remain the
existing implementation. check_runtime.py compares a real step-3,000 recovery
export with the original loader on one original source clip, CPU only. Its output
is an implementation check, not a WER estimate or the final model evaluation.

bind_whole_word.py admits only the verified step-6,000 export after complete local
backup and independent record recount. A model binding does not launch inference.
The source/destination runner and independent verifier are implemented. An actual
CPU execution on the step-3,000 recovery state completed one original source clip,
independent edit recount and a fresh replay with exactly matching tokens, text and
score. Its first attempt failed before inference because a locked video-loader
file was absent from the staged runtime; that attempt is retained. The helper was
restored with its original hash, and the separate second attempt passed. No
training or scientific inference implementation was changed by that repair.

The destination data check preserves all 768/1,167 assignments and successfully
decodes original grayscale pixels. All 1,935 destination media had already passed
the full staging hash and pixel check. The decoder selector reproduces both
archived B0 and FM development selections exactly, with 8,448 independent error
checks per model, and rejects six malformed inputs. This tests implementation;
it does not select a decoder or establish performance for a new checkpoint.

Per-device probes and measured-cost admission remain required before a new full
assessment. Existing training and source-assessment jobs take priority. The final
checkpoint binding, full-population collection and analysis are not completed by
the CPU fixture or by transferring files.

If conventional whole-word mixing explains the fixed-time gain, revise the claim
accordingly. If a fixed-time advantage remains, the separately prepared native
versus component-normalized comparison tests the loss-reduction explanation.
Do not broaden to more architectures or corpora before resolving these controls.
