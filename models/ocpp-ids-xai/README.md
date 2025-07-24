# How to execute:

First of all, make sure you have the Balanced_OCPP16_APP and Balanced_OCPP16_TCP-IP layer zips on datasets-zips folder.
Then, execute:

```bash
chmod +x configure_ids.sh
./configure_ids.sh
```
If there isn't a folder with a venv in the root of the project, it will create a virtual environment named `ocpp-venv` and install the requirements from `requirements.txt`.
If the folder already exists, it will just install the requirements.

After this, to run all experiments automatically:
```bash
chmod +x run_XAI.sh
./run_XAI.sh
```

## Without any type of XAI:
```bash
python3 client_xai.py --config conf/client_X/ocppflowmeter_config.json
```

## With SHAP:
```bash
python3 explainable_client.py config_client1.yaml --xai shap
```

## With LIME:
```bash
python3 explainable_client.py config_client1.yaml --xai lime
```

## With Both:
```bash
python3 explainable_client.py config_client1.yaml --xai both
```