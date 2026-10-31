# Research code for Fragment-Mixed Continuation

This package contains the scientific source behind the anonymous manuscript:
original FC/FM/FM-ST/CP training, source preparation and timing, whole-utterance
inference and decoder selection, matched full/fixed-time/whole-word/short-only
continuation, token-normalized losses, source retention, paired statistics,
exposure analysis, table export and figure generation. `Project/` retains the
original relative paths so that executed-file hashes remain meaningful.

The original training code identity is
`7e10ae375307c2e707c269d4c916c4940cedf87c1115ca553fb210a323d8240a`.
The six matched native/normalized/short-only driver identities are recorded in
`Metadata/Training_Identities.json`. The separate native ViCocktail reference
retains its own 101-file graph identity; it is not silently replaced by the
continuation recognizer. The source manifest records hashes and functional groups.

## Quick verification

Use Python 3.11 or newer for the release verifier. With NumPy installed and the
paper's `Data.zip` extracted as a neighboring `Data/` directory, run:

```text
python verify_release.py --data ../Data --output Verification_01
```

The destination must be new. This verifies archived file bytes, Python syntax,
the original training identity, all six matched driver identities and the
native-reference graph, then recomputes 60 aggregate rows and 49 intervals.
This numerical run needs no GPU, model weights, media or network access.

To also run the 27 original CPU unit tests, provide an interpreter containing
Torch, NumPy, SentencePiece and the preparation dependencies:

```text
python verify_release.py --data ../Data --test-python /path/to/python --output Verification_02
```

The observed unit-test environment used Python 3.9.13, Torch 2.8.0+cpu,
NumPy 2.0.2 and SentencePiece 0.2.2. The main numerical verifier used Python
3.11.9 and NumPy 2.0.2. The archived modern training/inference drivers require
Python 3.11+; the executed Tailscale assessment uses Python 3.12,
Torch 2.8.0+cu128, NumPy 1.26.4, FP32 and TF32 disabled. Do not infer
cross-version numerical equivalence from the CPU unit tests alone.

## Reproduction levels

`REPRODUCE.md` maps paper claims to code and describes neural prerequisites.
This distribution supplies the executed scientific code, tokenizer assets,
frozen plans/schedules and provenance dependencies. It does not redistribute
the video corpora or large model/optimizer checkpoints. `External_Assets.json`
indexes referenced external files; the registries, plans and model-identity
tables supply their locked identities. Provider terms continue to apply.

A fresh neural training run also needs real input preparation, resource checks
and a fresh finite execution admission. Historical admission deadlines and
hash checks are retained. They must not be disabled to make an unverified run
look equivalent. None of the CPU numerical/unit checks constitutes a fresh
6,000-update training replication.

Historical shared modules contain other methods and exploratory branches.
Only the cases mapped in `REPRODUCE.md` support the reported paper. Preparation
status strings in immutable plans describe when they were written; completed
results are identified by the accompanying paper, data and terminal receipts.
Cloud account access and paid-server management are outside the release workflow.

Original project code retains its existing license notices. Upstream
Auto-AVSR/ESPnet files and tokenizer assets retain their provenance and terms;
no blanket relicensing of third-party code or data is asserted here. Keep these
notices with any reuse. The numerical package has its own MIT license and the
derived measurements have CC BY 4.0 terms, as documented with the paper.
