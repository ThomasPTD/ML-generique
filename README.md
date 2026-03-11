🤖 Dockerized ML Pipeline

End-to-end Machine Learning training and inference pipeline, containerized with Docker. Supports any CSV input.

🚀 Quick Start

**# 1. Clone the repository
**git clone <repo-url>
cd ML_Pipeline

# 2. Copy your CSV into the data/ folder
cp your_file.csv data/

# 3. Launch Docker
docker-compose up --build

# 4. Access interfaces
#   → Jupyter: http://localhost:8888
#   → Streamlit: http://localhost:8501


📂 Project Structure

ML_Pipeline/
├── data/               ← Your CSV files (git-ignored)
├── notebooks/
│   └── 01_train_models.ipynb   ← Training notebook
├── models/             ← Saved models (git-ignored)
├── app/
│   └── main.py         ← Streamlit application
├── Dockerfile
├── docker-compose.yml
└── requirements.txt

⚙️ Features
📓 Jupyter Notebook (localhost:8888)
-Dynamic loading: Load CSV files via input widgets.
-Interactive selection: Choose features/target columns using checkboxes.
-Auto-detection: Automatically identifies Classification or Regression tasks.
-5 Algorithms trained and compared:
   Random Forest
   Gradient Boosting
   XGBoost
   SVM
   MLP (Neural Network)
-5-fold Cross-validation for every model.Metrics: Accuracy, Recall, F1, Confusion Matrix (Classif) / $R^2$, RMSE, MAE (Regression).
-Seaborn Visualizations: Barplots, CV boxplots, confusion matrices, and learning curves.
-Auto-save: The best-performing model is automatically exported.

🌐 Streamlit (localhost:8501)
-Single Prediction: Form dynamically generated based on model features.
-Batch Prediction: Upload a CSV file → Download processed results.
-Sidebar: Model summary and full benchmark report.

🛠️ Tech Stack

| Rôle | Technology |
|---|---|
| Containerization | Docker + Docker Compose |
| Notebook | Jupyter + ipywidgets |
| Interface | Streamlit |
| ML | scikit-learn, XGBoost |
| Visualisation | Seaborn, Matplotlib |
