#!/bin/bash

export TF_CPP_MIN_LOG_LEVEL='2' # Supress warnings from TensorFlow
export PYTHONPATH=$PYTHONPATH:$(pwd)/src

pkill python3

echo ""
echo "--- dataset 1: combined ocppflowmeter Dataset (SHAP + LIME) ---"

python3 -m src.server --config 'config/server/ocppflowmeter_config.json' &
echo "Waiting for server to start..."
sleep 10
echo "Starting clients for ocppflowmeter with LIME & SHAP..."
python3 -m src.client 'config/experiments/baseline/ocppflowmeter_config.json' --xai both &
wait
echo "--- OCPPFlowMeter Federation Finished ---"

sleep 5


echo ""
echo "--- dataset 2: cicflowmeter Dataset (SHAP + LIME) ---"
python3 -m src.server --config 'config/server/cicflowmeter_config.json' &
echo "Waiting for server to start..."
sleep 10
echo "Starting clients for cicflowmeter with LIME & SHAP..."
python3 -m src.client 'config/experiments/baseline/cicflowmeter_config.json' --xai both &
wait
echo "--- CICFlowMeter Federation Finished ---"

echo ""
echo "All experiments completed."
