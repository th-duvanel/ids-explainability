#!/bin/bash

DATASETS=("ocppflowmeter" "cicflowmeter")
EXPERIMENT_TYPES=("baseline" "modified_lime" "modified_shap" "modified_lime_shap")

export TF_CPP_MIN_LOG_LEVEL='2' # Suppress TensorFlow warnings
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "================================================="
echo "STARTING ALL EXPERIMENTS FOR THE TCC"
echo "================================================="

for dataset in "${DATASETS[@]}"; do
    for exp_type in "${EXPERIMENT_TYPES[@]}"; do
        
        echo ""
        echo "-------------------------------------------------"
        echo "RUNNING EXPERIMENT: Dataset=[$dataset] | Type=[$exp_type]"
        echo "-------------------------------------------------"

        echo "Terminating any existing Python processes..."
        pkill -f "python3 -m src"
        sleep 3

        SERVER_CONFIG="config/server/${dataset}/${exp_type}_config.json"
        CLIENT_CONFIG="config/experiments/${exp_type}/${dataset}_config.json"

        if [ ! -f "$SERVER_CONFIG" ] || [ ! -f "$CLIENT_CONFIG" ]; then
            echo "WARNING: Config files not found for this combination. Skipping."
            echo "Checked for server: $SERVER_CONFIG"
            echo "Checked for client: $CLIENT_CONFIG"
            continue
        fi

        echo "Starting server for [$dataset]..."
        python3 -m src.server --config "$SERVER_CONFIG" &
        SERVER_PID=$!
        
        echo "Waiting for server to initialize..."
        sleep 10

        echo "Starting client for experiment [$exp_type] on [$dataset]..."
        python3 -m src.client "$CLIENT_CONFIG" --xai both &
        CLIENT_PID=$!

        wait $CLIENT_PID
        echo "--- Client for [$dataset] / [$exp_type] has finished. ---"

        echo "Terminating server (PID: $SERVER_PID)..."
        kill $SERVER_PID
        wait $SERVER_PID 2>/dev/null

        sleep 5
    done
done

echo ""
echo "================================================="
echo "ALL EXPERIMENTS COMPLETED SUCCESSFULLY!"
echo "================================================="
