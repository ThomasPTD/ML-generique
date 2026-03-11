# 🤖 ML Pipeline Dockerisé

Pipeline complet d'entraînement et de prédiction Machine Learning, conteneurisé avec Docker. Supporte n'importe quel fichier CSV.

## 🚀 Démarrage rapide

```bash
# 1. Cloner le projet
git clone <url-du-repo>
cd ML_Pipeline

# 2. Copier votre CSV dans le dossier data/
cp votre_fichier.csv data/

# 3. Lancer Docker
docker-compose up --build

# 4. Accéder aux interfaces
#   → Jupyter : http://localhost:8888
#   → Streamlit : http://localhost:8501
```

## 📂 Structure du projet

```
ML_Pipeline/
├── data/               ← Vos fichiers CSV (non versionnés)
├── notebooks/
│   └── 01_train_models.ipynb   ← Notebook d'entraînement
├── models/             ← Modèles sauvegardés (non versionnés)
├── app/
│   └── main.py         ← Application Streamlit
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## ⚙️ Fonctionnalités

### 📓 Jupyter Notebook (`localhost:8888`)
1. **Chargement dynamique** du CSV via un widget de saisie
2. **Sélection interactive** des colonnes à utiliser (checkboxes)
3. **Détection automatique** : Classification ou Régression
4. **5 Algorithmes** entraînés et comparés :
   - Random Forest
   - Gradient Boosting
   - XGBoost
   - SVM
   - MLP (Réseau de Neurones)
5. **Cross-validation 5-fold** pour chaque modèle
6. **Métriques** : Accuracy, Recall, F1, Matrice de Confusion (Classif) / R², RMSE, MAE (Régression)
7. **Visualisations Seaborn** : barplots, boxplots CV, matrices de confusion, courbes d'apprentissage
8. **Sauvegarde automatique** du meilleur modèle

### 🌐 Streamlit (`localhost:8501`)
- **Prédiction unitaire** : formulaire généré dynamiquement selon les features du modèle
- **Prédiction en lot** : upload d'un CSV → téléchargement des résultats
- **Sidebar** : résumé du modèle et benchmark complet

## 🛠️ Technologies

| Rôle | Technologie |
|---|---|
| Conteneurisation | Docker + Docker Compose |
| Notebook | Jupyter + ipywidgets |
| Interface | Streamlit |
| ML | scikit-learn, XGBoost |
| Visualisation | Seaborn, Matplotlib |
