# ExoHunter-ML

### Machine Learning Pipeline for Automated Exoplanet Candidate Detection from Astronomical Light Curves

End-to-end Machine Learning project for the automated identification and classification of exoplanet candidates from astronomical light curves.

The project investigates the application of classical Machine Learning and Deep Learning techniques to astronomical time-series data, covering the complete Machine Learning lifecycle: data ingestion, exploratory analysis, preprocessing, feature engineering, model training, hyperparameter optimization, evaluation, explainability, inference, and production monitoring.

Developed as a project for the **Machine Learning Engineering** course, with emphasis on reproducibility, experimental validation, interpretability, and operationalization.

---

# 🌌 Overview

The discovery of exoplanets generates large volumes of astronomical observations that must be analyzed to identify subtle variations in stellar brightness.

One of the most important techniques used for exoplanet detection is the **transit method**.

When a planet passes in front of its host star relative to the observer, it causes a small decrease in the observed stellar brightness. This variation produces a characteristic signal in the star's **light curve**.

The objective of ExoHunter-ML is to investigate whether Machine Learning models can automatically identify these patterns and distinguish potential exoplanet candidates from non-planetary signals.

The project is designed not only as a predictive model, but as a complete and reproducible **Machine Learning engineering pipeline**.

---

# 🔭 Scientific Context

An astronomical light curve can be represented as a temporal sequence:

```text
Time
  │
  │     Stellar brightness
  │
  │ ─────────────────────────────
  │
  │              ↓
  │          Transit event
  │
  └──────────────────────────────
````

A transit event produces a characteristic decrease in the observed brightness of the star.

The system attempts to learn the statistical patterns associated with these events.

Conceptually:

```text
Astronomical Observation
          │
          ▼
     Light Curve
          │
          ▼
   Data Preprocessing
          │
          ▼
  Feature Engineering
          │
          ▼
   Machine Learning
          │
          ▼
   Candidate Classification
          │
          ▼
 Probability / Prediction
```

---

# 🎯 Objectives

## General Objective

Develop a complete and reproducible Machine Learning pipeline for the identification of exoplanet candidates using astronomical light curves.

## Specific Objectives

* Obtain and document a public astronomical dataset;
* Perform exploratory data analysis;
* Investigate the structure and quality of the observations;
* Handle missing, noisy, or inconsistent data;
* Develop preprocessing procedures;
* Investigate relevant features of astronomical light curves;
* Establish a baseline Machine Learning model;
* Train and compare different predictive models;
* Investigate sequential Deep Learning approaches;
* Optimize model hyperparameters;
* Evaluate model performance using appropriate metrics;
* Analyze prediction errors and failure cases;
* Apply explainability techniques;
* Demonstrate inference on previously unseen observations;
* Design a production-oriented architecture;
* Define monitoring and model-drift strategies;
* Define retraining criteria;
* Ensure reproducibility of the complete pipeline.

---

# 🧠 Machine Learning Approach

The project will investigate multiple modeling approaches rather than assuming that a more complex model is automatically better.

The experimental pipeline is expected to include:

```text
                    Dataset
                       │
                       ▼
                      EDA
                       │
                       ▼
               Preprocessing
                       │
                       ▼
            Feature Engineering
                       │
             ┌─────────┴─────────┐
             ▼                   ▼
       Classical ML        Sequential Model
             │                   │
             ▼                   ▼
       Baseline / RF            LSTM
             │                   │
             └─────────┬─────────┘
                       ▼
                 Model Evaluation
                       │
                       ▼
                 Explainability
                       │
                       ▼
                 Final Model
                       │
                       ▼
                  Monitoring
```

The final model will be selected based on experimental evidence rather than model complexity alone.

---

# 🤖 Models

The exact model configuration will be defined after the dataset analysis.

Potential approaches include:

### Classical Machine Learning

* Logistic Regression
* Random Forest
* Support Vector Machine
* Gradient Boosting / XGBoost, when appropriate

### Deep Learning

* LSTM
* 1D Convolutional Neural Network
* Other sequential architectures, if justified by the experiments

A simpler model will initially be used as a **baseline**, allowing the performance of more sophisticated approaches to be evaluated objectively.

---

# 📊 Evaluation

The models will be evaluated using metrics appropriate for the classification problem.

Potential metrics include:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
* Precision-Recall AUC
* Confusion Matrix

Particular attention will be given to **Recall and Precision**, since an astronomical candidate detection system must consider both missed candidates and false positives.

The final evaluation will also investigate:

* class imbalance;
* error distribution;
* false positives;
* false negatives;
* model calibration;
* generalization to unseen observations.

---

# 🔍 Explainability

Model predictions will be analyzed using explainability techniques.

Depending on the final architecture, the project may use:

* SHAP;
* Feature Importance;
* Permutation Importance;
* temporal importance analysis;
* other model-specific interpretation techniques.

The objective is to understand which characteristics of the observations contribute most strongly to the model's predictions.

> Feature importance will be interpreted as predictive association and will not be presented as evidence of causal relationships.

---

# 🛰️ Dataset

The project will use a publicly available astronomical dataset containing observations suitable for exoplanet candidate detection.

The final dataset will be selected based on:

* public accessibility;
* reproducibility;
* scientific relevance;
* sufficient number of observations;
* suitable class labels;
* compatibility with the proposed Machine Learning pipeline.

Possible sources include NASA's astronomical data repositories and datasets derived from missions such as **Kepler** and **K2**.

### Dataset documentation

Once selected, this section will document:

| Property               | Description |
| ---------------------- | ----------- |
| Source                 | TBD         |
| Mission                | TBD         |
| Number of observations | TBD         |
| Input format           | TBD         |
| Target variable        | TBD         |
| Number of classes      | TBD         |
| Class distribution     | TBD         |
| Missing values         | TBD         |

The dataset will not be stored directly in the repository when its size makes version control impractical.

Instead, the repository will contain reproducible data-ingestion procedures.

---

# 🧪 Experimental Methodology

The experimental process will follow a controlled Machine Learning workflow.

```text
1. Data Ingestion
        ↓
2. Data Validation
        ↓
3. Exploratory Data Analysis
        ↓
4. Data Cleaning
        ↓
5. Preprocessing
        ↓
6. Feature Engineering
        ↓
7. Train / Validation / Test Split
        ↓
8. Baseline Model
        ↓
9. Model Training
        ↓
10. Hyperparameter Optimization
        ↓
11. Final Evaluation
        ↓
12. Explainability
        ↓
13. Inference Demonstration
        ↓
14. Monitoring Strategy
```

Data leakage will be explicitly avoided throughout the pipeline.

All transformations learned from the training data will be fitted exclusively on the appropriate training partition.

---

# 🏗️ Project Architecture

The repository is organized to separate data, experimentation, source code, models, documentation, and results.

```text
ExoHunter-ML/
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── external/
│
├── notebooks/
│   └── ExoHunter_ML.ipynb
│
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   ├── evaluation/
│   └── explainability/
│
├── models/
│
├── reports/
│   ├── figures/
│   └── tables/
│
├── docs/
│   └── projeto.tex
│
├── app/
│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```

---

# 📁 Repository Structure

### `data/`

Contains data-related resources.

```text
data/
├── raw/
├── processed/
└── external/
```

Raw and processed datasets will generally not be versioned when their size makes this impractical.

---

### `notebooks/`

Contains the Google Colab notebooks used for experimentation and demonstration.

```text
notebooks/
└── ExoHunter_ML.ipynb
```

The final notebook will contain the complete executed pipeline required for the academic submission.

---

### `src/`

Contains reusable project code.

```text
src/
├── data/
│   ├── ingestion.py
│   └── preprocessing.py
│
├── features/
│   └── engineering.py
│
├── models/
│   ├── baseline.py
│   ├── random_forest.py
│   └── lstm.py
│
├── evaluation/
│   └── metrics.py
│
└── explainability/
    └── shap_analysis.py
```

---

### `models/`

Stores trained model artifacts when appropriate.

Large binary files will not be committed directly to Git when this would compromise repository size or reproducibility.

---

### `reports/`

Contains generated figures and tables used in the final documentation.

```text
reports/
├── figures/
└── tables/
```

---

### `docs/`

Contains project documentation, including the LaTeX project document.

```text
docs/
└── projeto.tex
```

---

### `app/`

Reserved for future inference or deployment components.

A production-oriented API may be implemented depending on the final scope of the project.

---

# 🔬 Reproducibility

Reproducibility is a central requirement of the project.

The implementation will follow these principles:

* Public datasets;
* No machine-specific local paths;
* Explicit dependency management;
* Fixed random seeds whenever applicable;
* Deterministic preprocessing where possible;
* Versioned source code;
* Documented data sources;
* Reproducible data ingestion;
* Reproducible training configuration;
* Clear separation between training and evaluation data;
* Google Colab compatibility.

The final notebook should execute from beginning to end without requiring manual modification.

---

# 🚀 Operationalization

Although the academic implementation will primarily use Google Colab, the project will also propose a production-oriented architecture.

A possible deployment workflow is:

```text
                  New Observation
                         │
                         ▼
                  Data Ingestion
                         │
                         ▼
                 Data Validation
                         │
                         ▼
                  Preprocessing
                         │
                         ▼
                  ML Inference
                         │
                         ▼
              ┌──────────┴──────────┐
              ▼                     ▼
        Prediction             Probability
              │                     │
              └──────────┬──────────┘
                         ▼
                  Result Storage
                         │
                         ▼
                    Monitoring
```

---

# 📡 Model Monitoring

A production implementation would monitor both the data and the model.

Potential indicators include:

### Data Monitoring

* Missing values;
* Input distribution;
* Feature distribution;
* Class distribution;
* Data drift.

### Model Monitoring

* Accuracy;
* Precision;
* Recall;
* F1-score;
* False-positive rate;
* False-negative rate;
* Prediction confidence;
* Model drift;
* Concept drift.

### Infrastructure Monitoring

* Inference latency;
* Throughput;
* Memory consumption;
* CPU/GPU utilization;
* Error rate.

---

# 🔄 Retraining Strategy

The model should not be considered permanently valid after deployment.

A retraining process may be triggered by:

```text
Performance degradation
        OR
Significant data drift
        OR
Concept drift
        OR
Large volume of newly labeled observations
        ↓
      Retraining
        ↓
      Validation
        ↓
 Model comparison
        ↓
  Model promotion
```

The exact thresholds will be defined based on the final model and experimental results.

---

# 🧩 Technologies

The project is expected to use the following technologies:

| Technology         | Purpose                      |
| ------------------ | ---------------------------- |
| Python             | Main programming language    |
| NumPy              | Numerical computation        |
| Pandas             | Data manipulation            |
| SciPy              | Scientific computation       |
| Scikit-learn       | Classical Machine Learning   |
| TensorFlow / Keras | Deep Learning                |
| SHAP               | Explainability               |
| Matplotlib         | Visualization                |
| Seaborn            | Statistical visualization    |
| Google Colab       | Reproducible experimentation |
| Git / GitHub       | Version control              |
| LaTeX              | Technical documentation      |

Additional libraries may be added according to the final methodology.

---

# 📓 Academic Deliverables

The project is being developed according to the requirements of the Machine Learning Engineering course.

The final submission will include:

* Complete project report;
* Fully executed Google Colab notebook;
* Source code;
* Experimental results;
* Model evaluation;
* Explainability analysis;
* Production monitoring strategy;
* Reproducibility documentation;
* Google Colab link;
* Presentation video.

---

# 📚 Documentation

The technical documentation is maintained in:

```text
docs/projeto.tex
```

The final report will contain:

1. Introduction
2. Problem Description
3. Dataset
4. Methodology
5. Machine Learning Pipeline
6. Experimental Results
7. Metrics Analysis
8. Model Explainability
9. Monitoring Strategy
10. Conclusion

---

# 📈 Project Status

🚧 **Under Development**

Current stage:

* [x] Project definition
* [x] Repository structure
* [x] Initial methodology
* [ ] Dataset selection
* [ ] Dataset ingestion
* [ ] Exploratory Data Analysis
* [ ] Preprocessing
* [ ] Feature Engineering
* [ ] Baseline
* [ ] Classical ML models
* [ ] LSTM model
* [ ] Hyperparameter optimization
* [ ] Model evaluation
* [ ] Explainability
* [ ] Inference demonstration
* [ ] Monitoring strategy
* [ ] Final documentation
* [ ] Presentation

---

# 🔭 Future Work

Potential extensions beyond the scope of the initial project include:

* Transformer-based time-series models;
* Attention mechanisms;
* Automated candidate ranking;
* Multi-mission datasets;
* Integration with additional astronomical catalogs;
* Real-time inference;
* Automated retraining;
* Active learning;
* Human-in-the-loop candidate validation;
* Deployment as an API;
* Integration with astronomical data services.

---

# ⚖️ Scientific and Ethical Considerations

The system developed in this project is intended as a computational aid for astronomical data analysis.

A Machine Learning prediction should not be interpreted as definitive scientific confirmation of an exoplanet.

A candidate identified by the model would require further astronomical validation and independent scientific analysis.

The project will distinguish between:

* model prediction;
* statistical evidence;
* scientific confirmation.

The use of public astronomical datasets and their respective licenses and attribution requirements will be documented.

---

# 📖 References

Key references and data sources will be documented as the project develops.

Potential sources include:

* NASA Exoplanet Archive;
* NASA Kepler Mission;
* K2 Mission;
* relevant astronomical literature;
* Scikit-learn documentation;
* TensorFlow/Keras documentation;
* SHAP documentation;
* Machine Learning literature.

---

# 👤 Author

**Jay Pitchula**

Engineering / Machine Learning / Aerospace

GitHub:

[@el-pitchula](https://github.com/el-pitchula)

---

# 📄 License

This project is distributed under the MIT License.

See [`LICENSE`](LICENSE) for details.

```
```
