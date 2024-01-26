#!/bin/bash
# Parallel Jobs
N_JOBS=10
ARGS="-P$N_JOBS --header :"

# Uncomment this line for dry run
#ARGS="--dry-run "$ARGS

# Experiment parameters
PROJECT=('CNN_Experiments')

# PROBLEMS=("Screw" "Sheet_Metal_Package" "Winding_Head" "Cable" "Cover")
PROBLEMS=("Screw" "Sheet_Metal_Package" "Winding_Head")

MAX_EPOCHS=(20 30 40)

LR=(0.0001 0.00001)

BATCH_SIZE=(32)

parallel $ARGS \
    sbatch \
        --job-name={project}_{problem} \
        $(echo --export=problem={problem},\
        epochs={max_epochs},lr={lr},batch={batch} | tr -d '[:space:]')\
        run-job.sh \
            ::: max_epochs "${MAX_EPOCHS[@]}" \
            ::: problem "${PROBLEMS[@]}" \
            ::: project "${PROJECT[@]}" \
            ::: lr "${LR[@]}" \
            ::: batch "${BATCH_SIZE[@]}"\
