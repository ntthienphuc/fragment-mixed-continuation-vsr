#!/usr/bin/env bash

WORKSPACE=/root/workspace
PROJECT="$WORKSPACE/project/Lip_Final"

source "$WORKSPACE/envs/pavir/bin/activate"
export PYTHONPATH="$PROJECT/baselines/auto_avsr${PYTHONPATH:+:$PYTHONPATH}"
export VSR_SPM_MODEL="$PROJECT/benchmark_v1/vocabulary/vi_unigram_2000.model"
export VSR_SPM_UNITS="$PROJECT/benchmark_v1/vocabulary/vi_unigram_2000.auto_avsr_units.txt"
export VSR_CACHE_STRICT=1
export VSR_VISUAL_ONLY=1

cd "$PROJECT"
