#!/bin/bash

DATASETS_ZIPS_FOLDER="./datasets-zips"
DATASETS_FOLDER="./datasets"

if [[ ! -d "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_APP_Layer" || ! -d "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_TCP-IP_Layer" ]]; then
    echo "Unzipping source datasets..."
    unzip -q "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_APP_Layer.zip" -d "$DATASETS_ZIPS_FOLDER"
    unzip -q "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_TCP-IP_Layer.zip" -d "$DATASETS_ZIPS_FOLDER"

    echo "Organizing datasets for clients..."
    for i in 1 2 3; do
        CLIENT_DIR="Client_${i}"
        mkdir -p "$DATASETS_FOLDER/$CLIENT_DIR/ocppflowmeter/"
        mkdir -p "$DATASETS_FOLDER/$CLIENT_DIR/cicflowmeter/"

        cp -r "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_APP_Layer/$CLIENT_DIR/." "$DATASETS_FOLDER/$CLIENT_DIR/ocppflowmeter/"
        cp -r "$DATASETS_ZIPS_FOLDER/Balanced_OCPP16_TCP-IP_Layer/$CLIENT_DIR/." "$DATASETS_FOLDER/$CLIENT_DIR/cicflowmeter/"
    done
    echo "Datasets organized."
fi

# Install and activate the virtual environment
VENV_DIR="ocpp-ids-venv"

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment in $VENV_DIR..."
    python3 -m venv "$VENV_DIR"
else
    echo "Virtual environment $VENV_DIR already exists."
fi

source "$VENV_DIR/bin/activate"

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
else
    echo "Warning: requirements.txt not found."
fi

echo "Setup complete. Virtual environment is active."