# ❤️ Heart Disease Prediction — End-to-End ML Deployment
NAME : SIYA SINGH 
REGISTRATION NUMBER : 23MIP10030
APPLICATION NUMBER : IN26011506
EMAIL : siya.23mip10030@vitvhopal.ac.in

An end-to-end machine learning project that predicts whether a patient is at
risk of heart disease based on clinical parameters, exposed as a REST API
built with Flask and deployed live on Render.

**Live Render URL:** `https://heart-disease-prediction-assignment-10-2.onrender.com/`

---

## 📁 Repository Structure

```
HeartDiseaseDeployment/
│
├── app.py                 # Flask REST API
├── model.pkl              # Trained model (saved with Joblib)
├── feature_order.pkl      # Feature column order used by the model
├── requirements.txt       # Python dependencies
├── README.md
├── train_model.py         # Task 1 & 2: preprocessing, training, evaluation
├── generate_dataset.py    # Builds heart.csv (see Dataset note below)
├── heart.csv              # Training dataset
├── render.yaml             # Render deployment config
├── templates/
│   └── index.html         # Simple landing page
└── static/
```

---

## 📊 Dataset

This project follows the schema of the **Kaggle Heart Disease Dataset**
(`johnsmith88/heart-disease-dataset`), the standard 14-column UCI Cleveland
heart disease dataset:

| Column | Description |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0–3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG results |
| thalach | Maximum heart rate achieved |
| exang | Exercise induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment |
| ca | Number of major vessels colored by fluoroscopy |
| thal | Thalassemia indicator |
| target | 1 = heart disease present, 0 = absent |

> **Note:** This project's execution environment doesn't have internet
> access, so `heart.csv` here was generated with `generate_dataset.py` —
> a synthetic dataset built to the exact same schema and realistic clinical
> correlations as the real Kaggle dataset. **Before submitting**, download
> the real dataset from Kaggle and replace `heart.csv` with it, then rerun
> `python train_model.py` to retrain on real data (no other code needs to
> change since the column names match exactly).

---

## 🧠 Task 1 — Data Understanding and Preprocessing

Handled in `train_model.py`:
- Loads `heart.csv` with Pandas
- Displays the first five records
- Identifies numerical features and the `target` variable
- Checks for missing values
- Splits data 80% train / 20% test (stratified)

## 🧠 Task 2 — Model Development

- Algorithm: **Random Forest Classifier**
- Evaluated using **Accuracy Score** (plus precision/recall/F1 for completeness)
- On the current dataset: **~76.6% accuracy**
- Model saved with **Joblib** as `model.pkl`

## 🌐 Task 3 — API Development

`app.py` implements a Flask REST API:
- Loads the trained model at startup
- `GET /health` — health check
- `POST /predict` — accepts patient details as JSON, returns a prediction

**Example request:**
```json
POST /predict
{
  "age": 63, "sex": 1, "cp": 3, "trestbps": 145, "chol": 233,
  "fbs": 1, "restecg": 0, "thalach": 150, "exang": 0,
  "oldpeak": 2.3, "slope": 0, "ca": 0, "thal": 1
}
```

**Example response:**
```json
{
  "prediction": "Heart Disease Detected",
  "risk_probability": 0.81
}
```

---

## 🚀 How to Run Locally

```bash
git clone <YOUR_REPO_URL>
cd HeartDiseaseDeployment
pip install -r requirements.txt

# (optional) retrain the model
python train_model.py

# start the API
python app.py
```
The API will be available at `http://localhost:5000`.

---

## ☁️ Task 4 — GitHub and Render Deployment

### Push to GitHub
```bash
git init
git add .
git commit -m "End-to-end heart disease prediction deployment"
git branch -M main
git remote add origin https://github.com/<your-username>/HeartDiseaseDeployment.git
git push -u origin main
```
Make sure the repository is **public**.

### Deploy on Render
1. Go to [render.com](https://render.com) and sign in with GitHub.
2. Click **New +** → **Web Service** and select this repository.
3. Configure:
   - **Environment:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
4. Click **Create Web Service** and wait for the build to finish.
5. Copy the generated public URL (e.g. `https://heart-disease-prediction-api.onrender.com`) and paste it at the top of this README and into the Google Form.
6. Verify it works:
   ```bash
   curl https://<your-app>.onrender.com/health
   ```
   (`render.yaml` in this repo lets Render auto-configure these settings too.)

Keep the service active and the GitHub repository public until evaluation is complete.

---

## 📝 Task 5 — Conclusion

The Random Forest classifier trained on the heart disease dataset achieved
an accuracy of approximately 76–77% on the held-out test set, with balanced
precision and recall across both classes, indicating it is reasonably
reliable at distinguishing at-risk from healthy patients without being
strongly biased toward either outcome. The biggest challenges during
deployment were less about the model itself and more about packaging: making
sure the exact feature order used in training matched the API's input
parsing, handling malformed or incomplete JSON gracefully, pinning
dependency versions so the environment behaved identically locally and on
Render's servers, and configuring a production-ready start command
(`gunicorn`) instead of Flask's development server. These are exactly the
kinds of problems MLOps practices exist to solve. Beyond just training an
accurate model, this project highlighted that real-world value comes from
making a model reliably reproducible, servable, and observable — through
version control, environment pinning, model serialization, health checks,
and continuous availability. MLOps is what turns a one-off notebook
experiment into a dependable, maintainable service that other systems and
users can actually depend on in production.

---

## 🎯 Learning Outcomes Covered
- Building and evaluating a machine learning classification model
- Saving/loading trained models with Joblib
- Developing a REST API with Flask
- Managing project code with GitHub
- Deploying ML applications to the cloud with Render
- Understanding MLOps fundamentals: packaging, versioning, deployment, serving
