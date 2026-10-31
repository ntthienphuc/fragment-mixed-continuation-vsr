"""Order the evidence around transfer, context, and source performance."""

from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "Source"

SOURCE_RESULT = r"""\subsection{Source and destination performance at the same fitted states}
\label{sec:source}
To determine whether the English transfer outcome accompanies a loss on the
continuation source, we assess the initializer, FC and FM on all 1,243 LRS2
test clips. This is a retrospective follow-up using the same final checkpoints
and readouts as the destination comparison, without source-test decoder tuning.
The common decoder gives WERs of 56.577, 50.976 and 52.432, respectively
(Table~\ref{tab:source}; Figure~\ref{fig:source}). Both continuation strategies
improve on the initializer. FM nevertheless produces 97 more errors than FC:
32 substitutions, 26 deletions and 39 insertions across 6,660 reference words.

\input{tables/source_profile}

The paired FC--FM difference is $-1.456$ WER points, with a 95\%
source-group bootstrap interval of [$-2.375$, $-0.584$]. Under the
development-selected readouts, FC/FM WER is 50.976/52.883 and the difference
is $-1.907$ points, with an interval of [$-2.986$, $-0.831$]. The English
source disadvantage is therefore present under either decoding comparison.
It is a disadvantage relative to full-clip continuation, not forgetting
relative to the initializer; LRS2 is the continuation source and this
experiment does not measure the initializer's upstream LRS3 performance.

\begin{figure}[tbp]
 \centering
 \includegraphics[width=\linewidth]{source_destination_profile.pdf}
 \caption{English source/destination WER with identical fitted states and a
 common decoder. Both continuations improve LRS2 relative to the initializer,
 but FM retains less source improvement than FC and does not improve WildVSR
 relative to FC. Panels use different, explicitly labelled vertical scales
 to show the within-corpus differences; WERs are not averaged across corpora.}
 \label{fig:source}
\end{figure}

Together with the WildVSR result, this profile shows that FM gives up part of
FC's source improvement without a demonstrated destination benefit in this
setting. It does not establish the same source/destination relationship for
Vietnamese, nor identify why the two continuation settings behave differently.
"""


def main():
    path = SOURCE / "sections/results.tex"
    text = path.read_text(encoding="utf-8")
    header = r"\subsection{Whole-utterance transfer from VLR to ViCocktail}"
    introduction = r"""\subsection{Transfer under a common decoding rule}
The two corpus pairs produce different outcomes under the same within-setting
comparison. FM improves VLR-to-ViCocktail transfer by 3.410 and 3.170 WER points
at seeds 17 and 29. It produces no clear LRS2-to-WildVSR improvement: the
common-decoder FC--FM difference is $-0.067$ points, with an interval that
includes zero. Figure~\ref{fig:gains} and Table~\ref{tab:comparisons} summarize
these contrasts. We first examine the complete destination results, then
analyze the training exposure and the role of contextual information.

\subsubsection{VLR to ViCocktail}
"""
    assert header in text
    text = text.replace(header, introduction)
    text = text.replace("presents the main Vietnamese comparison", "presents the VLR-to-ViCocktail comparison")
    error_marker = r"\subsection{Error changes and utterance length}"
    exposure_marker = r"\subsection{Training exposure and the role of context}"
    english_marker = r"\subsection{Portability to English cross-corpus recognition}"
    before, rest = text.split(error_marker, 1)
    error_section, rest = rest.split(exposure_marker, 1)
    context_section, english = rest.split(english_marker, 1)
    first_controls = "FM-ST and CP remain near FM:"
    training_start = "The training regimes differ materially."
    english_main, tail = english.split(first_controls, 1)
    english_controls, english_interpretation = tail.split(training_start, 1)
    text = before + r"\subsubsection{LRS2 to WildVSR}" + "\n" + english_main
    text += training_start + english_interpretation
    text += "\n" + error_marker + error_section
    text += exposure_marker + context_section
    text += "\nIn the English setting, " + first_controls[0].lower() + first_controls[1:] + english_controls
    text = text.replace("In the English setting, fM-ST", "In the English setting, FM-ST")
    text += "\n" + SOURCE_RESULT
    path.write_text(text, encoding="utf-8")
    path = SOURCE / "sections/experiments.tex"
    text = path.read_text(encoding="utf-8")
    start = text.index(r"\subsection{Main study and cross-setting check}")
    stop = text.index("The Vietnamese setting continues")
    new = r"""\subsection{Corpora and recognition systems}
We evaluate the same continuation intervention in two corpus-transfer settings:
VLR to ViCocktail and LRS2 to WildVSR. Each setting has matched within-setting
FC/FM comparisons and two same-target context controls. Differences between
the corpus pairs include language, initialization, training coverage and
development-data availability; their comparison assesses portability rather
than isolating a single factor.

"""
    text = text[:start] + new + text[stop:]
    marker = r"\subsection{Continuation budget and controls}"
    note = r"""A source-performance follow-up evaluates the English initializer, FC and FM
on the complete 1,243-clip LRS2 test set, containing 6,660 reference words and
354 archived source identifiers. It uses the same fitted states, preprocessing
and decoders as the WildVSR comparison. This source assessment is analyzed
separately from destination transfer.

"""
    text = text.replace(marker, note + marker)
    text = text.replace("English resampling uses released clips because a",
                        "WildVSR resampling uses released clips because a")
    text = text.replace("The English intervals are", "The WildVSR intervals are")
    text = text.replace("over independent recording sources. In both settings, intervals condition on the",
                        "over independent recording sources. The LRS2 source follow-up instead resamples\n"
                        "its 354 archived source groups, which are not asserted to be verified speaker\n"
                        "identities. These resampling units differ across corpora. All intervals condition on the")
    text = text.replace("Source-domain retention at the same final checkpoints\n"
                        "and unobserved upstream training overlap remain separate questions.",
                        "The source-performance follow-up also uses previously observed test data.\n"
                        "Upstream training overlap is not independently established.")
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
