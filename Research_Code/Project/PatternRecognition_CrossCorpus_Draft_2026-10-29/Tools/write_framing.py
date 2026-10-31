"""Write the focused cross-corpus framing without adding unfinished experiments."""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "Source"

INTRODUCTION = r"""\section{Introduction}
Visual speech recognition (VSR) recovers linguistic content from visible mouth
movements. A recognizer deployed on a new video collection must cope with changes
in speakers, recording conditions and utterance content. Large-scale supervised
learning has substantially improved benchmark accuracy
\cite{shillingford2019large,ma2023autoavsr}, yet strong source-corpus performance
does not ensure reliable recognition on another corpus \cite{djilali2024wildvsr}.
This motivates a practical question: how should an existing recognizer continue
learning from its available source videos before whole-utterance deployment
under a corpus shift?

We examine the temporal scale of supervision as a training intervention.
Complete utterances provide extensive visual and linguistic context. Shorter
fragments provide fewer neighboring frames and shorter target sequences, which
can change the balance between local evidence and sequence prediction. These
changes need not improve recognition: a temporal cut can also remove visual
evidence for a boundary word that remains in the target. The useful question is
whether this alternative supervision improves complete transcripts on a different
corpus, and under which continuation conditions that benefit appears.

Fragment-mixed continuation (FM) alternates complete source utterances with
fixed-duration fragments labelled by their overlapping transcript units.
Only the training presentations change. The resulting model recognizes a complete
video through the same architecture and decoding procedure as the initializer.
FM introduces no additional source videos and requires no fragment selection,
timestamp estimation or prediction merging at inference. This separation between
training views and deployment inputs makes it applicable to existing
whole-utterance recognizers.

Our comparison uses full-clip continuation (FC) with the same initialization,
source-clip order and number of optimizer updates. A common decoder provides the
main comparison of fitted states, while development-selected decoders describe
the calibrated systems. Matching presentations does not match the number of
visual frames or target tokens; we quantify these exposures and retain the
recognizer's native objective. The experiment therefore measures the complete
training intervention under an update budget, rather than attributing its effect
in advance to an isolated regularization mechanism.

Two corpus pairs expose contrasting outcomes. On VLR-to-ViCocktail transfer,
FM reduces common-decoder WER by 3.41 and 3.17 percentage points across two
seeds. On LRS2-to-WildVSR transfer, it gives no clear aggregate gain. A source
assessment further shows that English FM improves over the initializer but
retains less of FC's improvement on LRS2. These outcomes distinguish a useful
continuation result from a universal recipe. The corpus pairs differ in their
initializers, training coverage, repetition and calibration, so their contrast
does not isolate language as the cause.

To investigate the role of contextual information, we retain the fragment targets
and either restore neighboring pixels or supply predictions informed by the
complete video. Neither control improves on FM in the Vietnamese comparison.
Their edit-error profiles show that improved visual coverage or fewer
substitutions can be offset by additional deletions. This motivates examining
the complete output sequence when assessing a change in training context.

The paper contributes three results:
\begin{itemize}
 \item A fixed-time, overlap-labelled continuation formulation for existing
 whole-utterance VSR systems, evaluated against full-clip continuation with
 matched source presentations and optimizer updates.
 \item A decoder-controlled comparison across two corpus pairs, including a
 two-seed transfer gain and a source/destination assessment that identifies
 where the recipe does not improve recognition.
 \item Same-target context controls and error/exposure analyses that characterize
 the behavior of fragment supervision without modifying deployment inference.
\end{itemize}
Temporal cropping and mixed-length training are established in lip reading
\cite{martinez2020lipreading,prajwal2022subword}. The contribution examined here
is the behavior of this continuation formulation under corpus transfer,
including its benefits and conditions of failure.
"""

DISCUSSION = r"""\section{Discussion}
\subsection{When fragment supervision is useful}
The two corpus pairs show that the supervision presented during continuation
can matter even when the architecture and deployment inputs are fixed.
FM improves VLR-to-ViCocktail transfer across both fitted seeds and all
20 observed recording groups. Most of its advantage comes from fewer deletions
and insertions. In contrast, English FM does not improve WildVSR WER and gives
up part of FC's improvement on LRS2. The practical finding is therefore a useful
continuation option in one transfer regime, accompanied by a concrete case in
which the same intervention is not preferable.

This distinction matters when selecting a continuation strategy. Source
recognition and destination recognition answer different questions, and their
WERs should be considered separately. The English results show that improving
over the initializer on the continuation source is insufficient to establish
a destination benefit. They also distinguish a disadvantage relative to FC
from a loss relative to the initializer. A corresponding source/destination
trade-off has not been established for Vietnamese.

FM has a practical advantage in deployment complexity: it retains the same
single whole-utterance recognition procedure. It is relevant when source videos
and an existing recognizer are available but additional inference modules or
destination-domain gradient updates are undesirable. The Vietnamese experiment
does use destination development data for decoder calibration; the shared-decoder
result separates the observed gain from model-specific calibration. With the
remaining error rates, assisted transcription with human correction is a
plausible use case, rather than unattended transcription requiring high accuracy.
The application is a motivation; user-level utility has not been measured.

\subsection{What changes when context is restored}
The context-preserving control retains the fragment targets while restoring
neighboring visual evidence. It covers almost all selected annotation intervals
and retains part of the Vietnamese gain over FC, but performs worse than FM.
Its reduction in substitutions is outweighed by additional deletions. This
same-target contrast shows that more visible context is not sufficient to
improve the tested continuation recipe. It does not prove that deliberately
incomplete boundary evidence is the cause of the gain: context and label
coverage change together.

Full-view self-teaching provides contextual information through a predictive
distribution instead of additional student input. At the tested temperature and
weight, it adds no clear benefit. The simpler FM intervention is therefore
preferable among these evaluated variants. Neither result establishes that all
forms of context restoration or distillation are ineffective.

FM couples shorter inputs, shorter supervision and native loss reduction.
The English setting also repeatedly revisits a smaller, length-selected source
pool, whereas the Vietnamese setting presents each selected source clip once.
These differences suggest that supervision exposure and continuation regime
deserve attention before attributing the contrast to linguistic structure.
The observed frame/token counts cannot determine gradient contributions or
compute efficiency. A matched whole-word crop comparison and a declared common
loss normalization are the direct tests needed to separate conventional
short-view augmentation from the effects of overlap-labelled boundaries.

\subsection{Scope and reproducibility}
The evidence concerns two corpus pairs within one recognizer family. The
Vietnamese FC/FM comparison includes two seeds; the English comparison and
context controls include one. The assessment sets have been used in earlier
experiments, and the timing labels are automatic. Bootstrap intervals describe
variation within the observed assessment populations conditional on fitted
states; they do not estimate variation across all possible training runs.
Independent corpus confirmation and a second recognizer family would test
the portability of the result more directly than additional analysis of the
same test sets.

The numerical supplement provides model identities, settings, aggregate results
and per-clip edit counts. Dataset and checkpoint access follows the original
providers' terms. Derived measurements and preparation instructions support
reproduction without treating third-party media as newly collected or
relicensing those assets.
"""

CONCLUSION = r"""\section{Conclusion}
Fragment-mixed continuation changes the temporal scope of training supervision
while preserving ordinary whole-utterance visual speech recognition.
With matched source presentations and optimizer updates, it reduces
VLR-to-ViCocktail WER by 3.41 and 3.17 points under a common decoder across
two seeds. Fewer deletions and insertions account for most of this improvement;
restoring visual context or full-view predictive guidance does not improve
the tested recipe. The same intervention yields no clear WildVSR transfer
gain and is worse than full-clip continuation on the LRS2 source, while both
remain better than the initializer there. These results establish that changing
supervision granularity can improve complete-utterance transfer, but its value
depends on the continuation setting. Joint attention to destination accuracy,
source performance and supervision exposure provides a concrete basis for
developing that training strategy further.
"""

ABSTRACT = r"""Cross-corpus visual speech recognition requires a trained model to
recognize complete utterances under changed recording and content distributions.
We study fragment-mixed continuation: alternating full source clips with
two-second, overlap-labelled fragments while keeping the architecture and
whole-utterance inference unchanged. Comparisons use the same initialization,
source-clip schedule and optimizer updates as full-clip continuation. Across two
corpus pairs, the intervention produces distinct outcomes. On 1,167 ViCocktail
test clips after VLR continuation, a common decoder gives WER reductions of
3.41 and 3.17 percentage points across two seeds, driven mainly by fewer
deletions and insertions. Restoring neighboring visual context or full-view
predictive guidance does not improve the fragment-mixed result. On 2,854 WildVSR
clips after LRS2 continuation, there is no clear aggregate benefit. A separate
assessment of all 1,243 LRS2 test clips shows that fragment mixing improves over
the initializer but is 1.46 WER points worse than full-clip continuation under
the common decoder. The results identify supervision granularity as a useful
continuation choice in the studied Vietnamese transfer regime, while showing
that the benefit does not carry over automatically to a different corpus pair.
"""


def main():
    for name, content in [("introduction", INTRODUCTION), ("discussion", DISCUSSION),
                          ("conclusion", CONCLUSION)]:
        (SOURCE / "sections" / (name + ".tex")).write_text(content, encoding="utf-8")
    path = SOURCE / "main.tex"
    text = path.read_text(encoding="utf-8")
    text = text.replace("Cross-Corpus Vietnamese Visual Speech Recognition",
                        "Cross-Corpus Visual Speech Recognition")
    first = text.index(r"\begin{abstract}") + len(r"\begin{abstract}")
    last = text.index(r"\end{abstract}")
    text = text[:first] + "\n" + ABSTRACT + text[last:]
    text = text.replace("Vietnamese visual speech recognition ", "Visual speech recognition ")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
