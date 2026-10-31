# A usable ordinary word-aligned augmentation control

This revision addresses the training-metadata counterexamples retained in
Readiness_01. It is specified before constructing its new schedule and uses no
pending assessment predictions. The running T2/T4 protocol is unchanged.

Preserve the archived seed-17 original-video identities and 48,000 presentation
slots. Even slots retain the original complete video and transcript. On odd
slots, consider contiguous positive-duration word sequences that intersect the
old crop in time. Words with zero duration outside the chosen sequence do not
invalidate it. Do not interpolate their unknown durations or remove a word from
inside an otherwise contiguous sequence.

For each candidate, round its first word start down and its last covered word
end up to the 25 fps grid. Require the frame interval to fit in the original
video and its native token sequence to admit a CTC path. Choose the candidate
minimizing |new_start - old_start| + |new_stop - old_stop|. Break ties by distance
from 50 frames, midpoint displacement, start frame and word indices, in that order.
This is nearest-boundary conventional augmentation, with no novel-method claim.
Unlike the first preparation, a span need not cover a midpoint lying in silence.

If no candidate is valid, use the exact original full video and full transcript.
Record the fallback and apply the same full-view fallback to a newly trained
matched T1 arm. Existing archived T1 is only an exact schedule reference when
the new fallback count is zero; otherwise it is a practical comparison with a
declared view difference. No sample is dropped. The ordinary full-view T0 control
already uses the original full view in every slot.

Report duration, target-token and pixel-interval changes for the same slots;
equal update count does not mean equal compute or loss scale. Frame rounding
can expose part of an adjacent unlabelled word and is not proof of a visually
clean speech boundary. Word timings are automatic. Full-view fallback preserves
coverage but does not establish timing accuracy.

Before training, independently verify schedules and tokenization, locate and
hash the actual training pixels, run real loss/gradient checks, and predeclare
source-development selection and source/destination evaluation. Train new arms
only when the completed T2/T4 results leave a useful mechanism question and a
finite resource budget has been admitted. This preparation admits no paid work.
