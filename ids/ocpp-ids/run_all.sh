#!/bin/bash

DATASETS=("ocppflowmeter" "cicflowmeter")
EXPERIMENT_TYPES=("lime" "shap")

export TF_CPP_MIN_LOG_LEVEL='2'
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

echo "================================================="
echo "STARTING SCALED EXPERIMENTS FOR THE TCC"
echo "================================================="

#echo "-------------------------------------------------"
#echo "RUNNING BASELINE EXPERIMENTS"
#echo "-------------------------------------------------"
#for dataset in "${DATASETS[@]}"; do
#    SERVER_CONFIG="config/server/${dataset}/baseline_config.json"
#    CLIENT_CONFIG="config/experiments/baseline/${dataset}_config.json"
#    
#    echo "Starting server for [${dataset}] baseline..."
#    python3 -m src.server --config "$SERVER_CONFIG" &
#    SERVER_PID=$!
#    sleep 5
#    
#    echo "Starting client for [${dataset}] baseline..."
#    python3 -m src.client "$CLIENT_CONFIG" --xai both &
#    CLIENT_PID=$!
#    wait $CLIENT_PID
#    
#    kill $SERVER_PID
#    wait $SERVER_PID 2>/dev/null
#    sleep 5
#done

for dataset in "${DATASETS[@]}"; do
    
    # Define os níveis de remoção para cada dataset
    if [ "$dataset" == "cicflowmeter" ]; then
        REMOVAL_COUNTS=(8 16 32 64)
    else # ocppflowmeter
        REMOVAL_COUNTS=(8 16 32)
    fi

    for exp_base_type in "${EXPERIMENT_TYPES[@]}"; do
        for count in "${REMOVAL_COUNTS[@]}"; do
            exp_type="${exp_base_type}_${count}"

            echo ""
            echo "-------------------------------------------------"
            echo "RUNNING: Dataset=[$dataset] | Type=[$exp_type]"
            echo "-------------------------------------------------"

            pkill -f "python3 -m src"
            sleep 2

            SERVER_CONFIG="config/server/${dataset}/${count}_config.json"
            CLIENT_CONFIG="config/experiments/${exp_type}/${dataset}_config.json"

            if [ ! -f "$SERVER_CONFIG" ] || [ ! -f "$CLIENT_CONFIG" ]; then
                echo "WARNING: Config files not found. Skipping."
                echo "Server: $SERVER_CONFIG"
                echo "Client: $CLIENT_CONFIG"
                continue
            fi

            echo "Starting server..."
            python3 -m src.server --config "$SERVER_CONFIG" &
            SERVER_PID=$!
            sleep 5

            echo "Starting client..."
            python3 -m src.client "$CLIENT_CONFIG" &
            CLIENT_PID=$!

            wait $CLIENT_PID
            echo "--- Client finished. ---"

            kill $SERVER_PID
            wait $SERVER_PID 2>/dev/null
            sleep 5
        done
    done
done

echo "================================================="
echo "ALL EXPERIMENTS COMPLETED!"
echo "================================================="