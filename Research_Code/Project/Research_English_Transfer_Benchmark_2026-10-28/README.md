# Whole-utterance English transfer benchmark

Updated 28 October 2026, 03:52 Vietnam time.

The experiment asks whether the existing full/crop adaptation recipe improves
complete-clip recognition after moving from LRS2 to WildVSR. It is a candidate
for a separate training/transfer paper. It does not change the interval paper.

The frozen comparison contains the actual shared initializer, T0 and T1 at
6,000 updates, and an independently trained stronger English reference. Each
model selects its decoder using all 1,082 LRS2 development clips; all selections
must close before evaluating any of the 2,854 WildVSR clips. A fixed native
decoder is a prespecified secondary readout. See `Protocol.md` for the design.

Complete: inputs and all 4,328 source feature packets checked; 132 native versus
cached source prediction comparisons passed with zero score difference. The
current execution also passed the 64-task coverage and fail-closed checks.

Running: source calibration on the local GPU under `Local_Queues/Local_01`,
with a two-hour limit ending at 04:56:40. The first two 136-clip B0 shards passed
independent verification; the third is running. Each completed shard took about
21 minutes including verification. These are 272 model/clip instances of 4,328.
No English assessment result or complete development WER is available yet.
The task queue preserves verified work and backs it up to C: between tasks.
The two-hour limit is a local allocation limit, not an ETA for the whole study.

The cloud continuation package is now prepared and checked. Its 7.54 GB payload
passed installation and exact native CPU replay for all four model states;
the transport, restoration and finite-stop helpers passed their operational
checks. No new English pod has been rented. After the local queue ends, freeze
its verified shards for reuse, check current balance and price, and admit the
remaining workload with a finite budget. See `Server_Operations/README.md`.

The existing English checkpoints were adapted on the same 7,291 eligible long
clips, not the entire LRS2 train set. A new read-only reconstruction verified all
48,000 sample decisions. See
`../Research_English_Training_Scope_2026-10-28/Findings.md` for the exposure,
duration and word-boundary implications. The benchmark protocol remains frozen.

`Plan.json` and `Execution_Lock.json` record the unchanged design and source
hashes. A complete claim requires every model and task, independent verification,
decoder selection, assessment analysis and a verified backup. Partial outputs
are reusable computation, not a completed comparison.
