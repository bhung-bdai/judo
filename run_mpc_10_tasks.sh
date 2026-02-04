#!/bin/bash
# Script to collect data on 10 Spot tasks using run_mpc.py
# with record_rollouts=False and cem optimizer

TASKS=(
    "spot_box_push"
    "spot_box_upright"
    "spot_chair_push"
    "spot_chair_upright"
    "spot_cone_push"
    "spot_cone_upright"
    "spot_rack_push"
    "spot_rack_upright"
    "spot_tire_push"
    "spot_tire_upright"
)

OPTIMIZER="cem"
RECORD_ROLLOUTS="false"
NUM_EPISODES=5

echo "Starting MPC data collection for 10 Spot tasks..."
echo "Optimizer: ${OPTIMIZER}"
echo "Record rollouts: ${RECORD_ROLLOUTS}"
echo "Number of episodes per task: ${NUM_EPISODES}"
echo ""

for TASK in "${TASKS[@]}"; do
    echo "========================================"
    echo "Running task: ${TASK}"
    echo "========================================"

    python run_mpc/run_mpc.py \
        --init-task ${TASK} \
        --init-optimizer ${OPTIMIZER} \
        --num-episodes ${NUM_EPISODES} \
        --no-record-rollouts \
        --no-visualize

    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed ${TASK}"
    else
        echo "✗ Error running ${TASK}"
    fi
    echo ""
done

echo "========================================"
echo "All tasks completed!"
echo "Results saved in: run_mpc/results/"
echo "========================================"
