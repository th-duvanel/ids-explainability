# Changelog



This project is a fork of the original OCPP IDS project, which is a IDS (Intrusion Detection System) for OCPP (Open Charge Point Protocol) networks.
We added the possibility to use XAI (Explainable Artificial Intelligence) techniques to explain the decisions made by the IDS and to test those decisions, changing the original code and datasets. This will compare the perfomance of the IDS with and without XAI techniques.

## 1. Integrated Explainable AI (XAI)

The client has been enhanced with two industry-standard XAI libraries to provide insights into the model's behavior at the local client level.

* **LIME (Local Interpretable Model-agnostic Explanations)**
    * **Purpose:** To explain individual predictions by creating a local, interpretable model. It answers the question: "Why did my model make this specific prediction for this one data sample?"
    * **Output:** Generates easy-to-read, interactive **HTML files** for each explanation.
    * **Implementation:** A `calculate_and_save_lime()` method was added. It is triggered during the `evaluate` step if requested.
    * **Example Output File:** `results/lime/lime_explanation_client_1_instance_0.html`

* **SHAP (SHapley Additive exPlanations)**
    * **Purpose:** To calculate the contribution of each feature to a specific prediction using a game-theoretic approach. It assigns a "Shapley value" to each feature, quantifying its impact.
    * **Output:** Saves the raw Shapley values as **NumPy array files (`.npy`)**. These files can be used for advanced offline analysis and for generating various plots (e.g., summary plots, force plots).
    * **Implementation:** A `calculate_and_save_shap()` method was added, which is also triggered during the `evaluate` step.
    * **Example Output File:** `results/shap/shap_values_client_1.npy`

## 2. Flexible Execution Control

To easily manage experiments without modifying the source code, a command-line argument has been added to control the execution of XAI methods.

* **Argument:** `--xai`
* **Functionality:** Allows the user to specify which, if any, XAI method to run during the client's execution.
* **Options:**
    * `shap`: Runs only the SHAP analysis.
    * `lime`: Runs only the LIME analysis.
    * `both`: Runs both SHAP and LIME.
    * `none`: (Default) Runs the standard federated learning process without any XAI calculations.

#### Usage Examples:

* **Run without XAI:**
    ```bash
    python3 explainable_client.py conf/client_1/config.yaml
    ```
    *or*
    ```bash
    python3 explainable_client.py conf/client_1/config.yaml --xai none
    ```

* **Run with only SHAP:**
    ```bash
    python3 explainable_client.py conf/client_1/config.yaml --xai shap
    ```

* **Run with only LIME:**
    ```bash
    python3 explainable_client.py conf/client_1/config.yaml --xai lime
    ```

* **Run with both SHAP and LIME:**
    ```bash
    python3 explainable_client.py conf/client_1/config.yaml --xai both
    ```

## 3. Automated Metrics Logging

To facilitate experiment comparison and track model performance over time, a robust metric logging system was implemented.

* **Purpose:** Automatically save key performance metrics from each evaluation round to a structured file. This is crucial for comparing the baseline model's performance against models trained on datasets modified based on XAI insights.
* **Output:** A separate **CSV file for each client** (e.g., `client_1_metrics.csv`). The script appends a new row to this file for every evaluation round, so results from multiple experiments are stored in one place.
* **Implementation:**
    1.  A new method `save_metrics_to_csv()` was added to the `ExplainableClient` class.
    2.  This method is called from `evaluate()` in every round.
    3.  The CSV file includes not only the metrics but also important contextual information.

#### CSV File Structure

The generated CSV files will have the following columns:

| Column          | Description                                                     |
| :-------------- | :-------------------------------------------------------------- |
| `timestamp`     | The date and time when the evaluation round occurred.           |
| `client_id`     | The ID of the client being evaluated.                           |
| `round`         | The current federated learning round number.                    |
| `xai_method_used`| The `--xai` option used for that run (`none`, `shap`, etc.).      |
| `accuracy`      | The accuracy score for the round.                               |
| `f1_score`      | The F1-score for the round.                                     |
| `...`           | Any other metrics returned by the `evaluation_metrics` function. |

## 4. Code and Configuration Requirements

To support these new features, some minor changes to helper files and configurations are required.

* **Configuration File (`config.yaml`)**
    * A new key, `results_output_dir`, must be added to specify the location for the metrics CSV files.
    * **Example:**
        ```yaml
        # ... other configurations
        shap_output_path: './results/shap'
        lime_output_path: './results/lime'
        
        # ADD THIS KEY
        results_output_dir: './results/metrics'
        ```

* **Metrics Function (`utils.py`)**
    * The `evaluation_metrics()` function must be modified to return a **dictionary** of metrics. This allows for a flexible and self-describing way to log results.
    * **Example:**
        ```python
        def evaluation_metrics(y_true, y_pred_classes, ...):
            # ... calculations for accuracy, f1, etc.
            
            # MUST return a dictionary
            return {
                'accuracy': accuracy,
                'f1_score': f1_score,
                'precision': precision,
                'recall': recall
            }
        ```

* **Data Preprocessing (`preprocess` function)**
    * The `preprocess` function in `explainable_client.py` was modified to return the `feature_names` from the dataset. This is essential for LIME and SHAP to provide human-readable outputs.

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