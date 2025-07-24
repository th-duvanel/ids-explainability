#!/bin/bash

# This script runs the federated learning process for two different datasets.
# For each client, it enables both SHAP and LIME explainability methods.

# Make sure the unified client script is named 'client_xai.py'

export TF_CPP_MIN_LOG_LEVEL=1

echo "--- dataset 1: ocppflowmeter Dataset (SHAP + LIME) ---"

# Start the server for the first dataset in the background
python3 -m server --config 'conf/server/ocppflowmeter_config.json' &

# Wait for the server to initialize
echo "Waiting for server to start..."
sleep 15

# Start clients for the first dataset, each with SHAP and LIME enabled
echo "Starting clients for ocppflowmeter..."
python3 -m client_xai 'conf/client_1/ocppflowmeter_config.json' --xai both &
python3 -m client_xai 'conf/client_2/ocppflowmeter_config.json' --xai both &
python3 -m client_xai 'conf/client_3/ocppflowmeter_config.json' --xai both &

# Wait for all client processes of the first experiment to finish
wait
echo "--- OCPPFlowMeter Federation Finished ---"

# Optional: A small pause between experiments
sleep 5

echo ""
echo "--- dataset 2: cicflowmeter Dataset (SHAP + LIME) ---"

# Start the server for the second dataset in the background
python3 -m server --config 'conf/server/cicflowmeter_config.json' &

# Wait for the server to initialize
echo "Waiting for server to start..."
sleep 15

# Start clients for the second dataset, each with SHAP and LIME enabled
echo "Starting clients for cicflowmeter..."
python3 -m client_xai 'conf/client_1/cicflowmeter_config.json' --xai both &
python3 -m client_xai 'conf/client_2/cicflowmeter_config.json' --xai both &
python3 -m client_xai 'conf/client_3/cicflowmeter_config.json' --xai both &

# Wait for all client processes of the second experiment to finish
wait
echo "--- CICFlowMeter Federation Finished ---"

echo ""
echo "All experiments completed."