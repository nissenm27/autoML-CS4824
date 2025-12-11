# AutoML-CS4824: Lightweight AutoML Framework
**Course:** CS4824 — Machine Learning Capstone  
**Author:** Matthew Nissen  
**Project:** Building a resource-conscious AutoML framework inspired by Auto-sklearn and Google AutoML  
**Status:** Week 2 — Preprocessing Module Implementation  

---

## Project Overview
This project aims to design and implement a lightweight AutoML system that automates the essential stages of the machine learning pipeline — **data preprocessing, model selection, and hyperparameter tuning** — while remaining transparent and computationally efficient.  
The framework will incrementally expand toward more advanced AutoML capabilities such as **Bayesian optimization, ensemble construction, and meta-learning**.

---

## Week 2: Progress Summary — Preprocessing Module Implementation

### Datasets
- Downloaded datasets via **OpenML API**:
  - `iris.csv`
  - `wine_quality.csv`
  - `adult_income.csv`
- Stored locally under `/data/` for consistent offline access and reproducibility.

### Schema Inference
- Implemented `infer_schema(df)` in `preprocessing/schema.py`:
  - Automatically detects numeric vs. categorical features.
  - Generates missingness summaries.
  - Returns structured metadata as a `Schema` dataclass instance.

### Preprocessing Pipeline
- Implemented `build_preprocessor(schema)` in `preprocessing/build_preprocessor.py`:
  - Handles **imputation** (median for numeric, most frequent for categorical).
  - Applies **scaling** (StandardScaler) and **encoding** (OneHotEncoder).
  - Designed for easy integration into the AutoML pipeline via scikit-learn’s `ColumnTransformer`.

### Validation
- Created `validate_preprocessing.py` to verify:
  - End-to-end compatibility of preprocessing with scikit-learn models.
  - Successful runs on Iris and Wine Quality datasets.
  - Output: printed accuracy for Iris classification, RMSE for Wine regression.

**Result:** Preprocessing modules are fully functional and validated on small datasets.  
This completes the Week 2 milestone defined in the project timeline.

---

## Week 3: Progress Summary — Model Wrapper Implementation

### Model Architecture
- Created a standardized **BaseModel** interface in `models/base_model.py` with unified methods:
  - `.train(X, y)` — fits the model  
  - `.predict(X)` — generates predictions  
  - `.score(X, y)` — computes task-specific metrics (**accuracy/F1** for classification, **RMSE** for regression)
- Ensures all models integrate seamlessly into future AutoML search pipelines.

### Linear Models
- Implemented in `models/linear_models.py`:
  - **Logistic Regression:** baseline classifier with interpretable decision boundaries  
  - **Ridge Regression:** regression benchmark for numeric prediction tasks  
- Serves as the baseline family for comparison against tree-based and neural models.

### Tree-Based Models
- Implemented in `models/tree_models.py`:
  - **Decision Tree:** interpretable single-tree model for fast experimentation  
  - **Random Forest:** ensemble method providing stability and reduced variance  
  - **Gradient Boosting:** boosting-based ensemble achieving highest baseline accuracy on validation data  
- All tree models share the same interface and integrate with the preprocessing pipeline.

### Neural Network
- Implemented in `models/neural_net.py`:
  - Simple **feed-forward MLP** supporting both classification and regression  
  - Architecture: two hidden layers (64, 32 neurons) with ReLU activations and a maximum of 500 iterations  
  - Provides a foundation for future deep-learning extensions (NAS, meta-learning)

### Validation
- Created `models/validate_models.py` to verify end-to-end compatibility of all wrappers with preprocessing modules  
- Validated on the **Iris** dataset using the unified preprocessing + model pipeline  

**Output Scores**
| Model | Accuracy |
|:--------------------|:-----------:|
| Logistic Regression | 0.857 |
| Decision Tree | 0.822 |
| Random Forest | 0.859 |
| Gradient Boosting | 0.874 |
| Neural Network | 0.840 |

**Result:** All model wrappers train and evaluate successfully through the preprocessing pipeline.  
A complete **Model Zoo** has been established, forming the backbone for Week 4’s hyperparameter search and AutoML pipeline integration.


## Week 4: Progress Summary — Baseline Search Strategies & Pipeline Integration

### Search Framework
- Developed a unified **SearchManager** in `search/grid_random.py` supporting both **Grid Search** and **Random Search**.  
- Enables hyperparameter optimization for any scikit-learn-compatible model within a preprocessing → model pipeline.  
- Accepts customizable arguments such as `scoring`, `cv`, `n_iter`, and `random_state` for flexible experimentation.

### Parameter Grids
- Defined task-specific parameter grids in `search/search_utils.py` for all supported models:  
  - **Logistic Regression:** solver + regularization strength  
  - **Ridge Regression:** alpha (regularization)  
  - **Decision Tree:** depth + split thresholds  
  - **Random Forest:** estimators + depth  
  - **Gradient Boosting:** learning rate + depth + estimators  
  - **Neural Network:** layer size + learning rate  
- Allows systematic exploration of each model’s hyperparameter space.

### AutoML Orchestrator
- Implemented the end-to-end **`automl_orchestrator.py`** module integrating:
  - Automatic dataset loading and preprocessing  
  - Model selection from the Model Zoo  
  - Hyperparameter search via Grid/Random Search  
  - Cross-validation evaluation and test-set scoring  
  - Leaderboard generation and ranking of tuned models  
- Added **automatic task-type detection** (`classification` vs `regression`) and skipping of incompatible models, enabling dataset-agnostic operation.

### Validation
- Executed the full AutoML pipeline on the **Iris** dataset (classification).  
- Confirmed correct task detection and model skipping (`Ridge` excluded automatically).  
- Each model completed cross-validated hyperparameter search and test-set evaluation.

**Output Leaderboard**
| Model | Best CV Score | Test Score | Key Parameters |
|:--------------------|:------------:|:------------:|:----------------|
| Gradient Boosting | 0.8716 | 0.8768 | `n_estimators=100`, `max_depth=4`, `learning_rate=0.2` |
| Decision Tree | 0.8553 | 0.8601 | `max_depth=10`, `min_samples_split=2` |
| Random Forest | 0.8564 | 0.8590 | `n_estimators=100`, `max_depth=10` |
| Logistic Regression | 0.8516 | 0.8520 | `C=1`, `solver='liblinear'` |
| Neural Net | 0.8449 | 0.8513 | `hidden_layer_sizes=(32,)`, `learning_rate_init=0.001` |

**Result:** The AutoML framework now performs full-cycle automation—data preprocessing, model training, and hyperparameter tuning—with leaderboard evaluation.  
This milestone completes **Week 4**, establishing a robust baseline AutoML system ready for Week 5’s enhancements in logging, efficiency analysis, and cross-dataset evaluation.

