# Lightweight AutoML for Efficient and Transparent Model Selection
**Author:** Matthew Nissen  
**Course:** CS 4824 — Machine Learning  
**Instructor:** Prof. Ming Jin  

This repository contains the full implementation of a lightweight, transparent AutoML framework designed for low-compute environments. The system automates:

- Data preprocessing (schema inference, imputation, scaling, encoding)
- Model selection across a curated model zoo
- Hyperparameter optimization (Grid Search, Random Search, Bayesian Optimization)
- Ensemble construction (soft voting + stacking)
- Meta-learning warm starts
- Lightweight neural architecture search (TinyNAS)

The framework is optimized for laptop-scale execution while maintaining transparency and interpretability.

---

## **Getting Started**

### **1. Install Dependencies**
You can install all required libraries using:

```bash
pip install -r requirements.txt
```

### **2. Repository Structure:**

```markdown
autoML-CS4824/
│
├── preprocessing/
│   ├── schema.py
│   ├── build_preprocessor.py
│   ├── validate_preprocessing.py
│
├── models/
│   ├── base_models.py
│   ├── linear_models.py
│   ├── tree_models.py
│   ├── neural_net.py
│   ├── validate_models.py
│
├── search/
│   ├── grid_random.py
│   ├── bayesian.py
│   ├── search_utils.py
│   ├── validate_search.py
│
├── ensembles/
│   ├── __init__.py
│   ├── soft_voting.py
│   ├── stacking.py
│
├── meta/
│   ├── meta_features.py
│   ├── meta_knn.py
│   ├── meta_dataset.json
│
├── nas/
│   ├── tiny_nas.py
│
├── results/
│   ├── model_zoo_summary.csv
│   ├── leaderboard_*.csv
│   ├── *_*_history.json
│   ├── model_zoo_bubble_plot.png
│
├── data/
│   ├── iris.csv
│   ├── wine_quality.csv
│   ├── adult_income.csv
│   ├── inspect_datasets.py
│   ├── load_datasets.py
│
├── notebooks/
│   ├── example_run.ipynb
│
├── reports/
│   ├── annotated-Project-Milestones.pdf
│   ├── AutoML_presentation.pdf
│   ├── Milestone Report 2.pdf
│   ├── CS4824_Final_Capstone_Report.pdf
│   ├── CS4824_Final_Capstone_Report.Rmd
│   ├── AutoML.docx
|
├── checkpoints/
│   ├── best_model_adult.pkl
│   ├── best_model_iris.pkl
│   ├── best_model_wine.pkl
|
├── analysis/
│   ├── build_meta_table.py
│   ├── model_zoo_plot.py
│   ├── plot_dataset_landscape.py
│   ├── plot_search_trajectories.py
│   ├── slide4_ensemble_and_warmstart.py
│   ├── dataset_landscape.png
│   ├── analysis/
|   │   ├── meta_table.csv
|   │   ├── slide4_ensemble_and_warmstart.png
│
├── run_all_datasets.py
|
├── requirements.txt
|
└── automl_orchestrator.py
```

### **3. Running AutoML:**
Run the full AutoML pipeline across all datasets (currently set to Bayesian Optimization):

```bash
python automl_orchestrator.py --all
```
This will run:

- Adult Income

- Iris

- Wine Quality

and produce:

- Leaderboards

- Search histories

- Updated meta-learning database

- Best model checkpoint per dataset


Or run a single dataset:
```bash
python automl_orchestrator.py --data data/iris.csv --target class
```

Outputs include:
- Leaderboard CSVs

- Model checkpoints

- Search histories

- Meta-learning database updates


### **4. Example Notebook:**
An included notebook:
```bash
notebooks/example_run.ipynb
```
demonstrates:
- Loading the datasets

- Running the AutoML system

- Plotting leaderboard comparisons

- Visualizing search trajectories

- Inspecting TinyNAS architectures

### **5. Model Checkpoints:**
Best models for each dataset are saved automatically to:
```bash
checkpoints/
    best_model_<dataset>.pkl
```

You can load a model with:
```python
import joblib
model = joblib.load("checkpoints/best_model_adult.pkl")
model.predict(X)
```

### **6. Academic Integrity Disclosure:**
This project used ChatGPT (OpenAI, 2025 version) for assistance in drafting documentation, refining explanations, and debugging certain components. All implementation, experiments, and analysis were performed by the author.

OpenAI. (2025). ChatGPT (Dec 2025 Model): Conversation-based assistance with analysis, editing, and explanation. 
Retrieved from https://chat.openai.com/


### **7. Future Work:**
Full list available in the final report. Key next steps:

- Expand metric suite (F1, AUC, RMSE, MAE)

- Improve regression support and detection

- Broaden model zoo (XGBoost, CatBoost, LightGBM)

- Explore deeper NAS and meta-learning