# Student Performance Analysis and Prediction Using AI

**Internship:** AICTE | IBM SkillsBuild Data Analytics with AI Internship 2026  
**Project type:** Educational data analytics and machine learning demonstration  
**Author:** Mohammad Saifullah (replace/update if your registered submission name differs)

## 1. Project overview
This project explores how study hours, attendance, previous scores, assignment completion and sleep hours relate to an illustrative final score. It uses charts and a Random Forest classifier to demonstrate how a model can classify a record into `High`, `Medium`, or `Needs Support`.

## 2. Important dataset note
`student_performance_dataset.csv` is a **synthetic, computer-generated demonstration dataset** containing 600 illustrative records. It is not collected from real students and should not be used to make real educational decisions. The dataset is included in this project folder, so no download is needed.

## 3. Tools and libraries
- Python
- Pandas and NumPy for data handling
- Matplotlib for charts
- Scikit-learn for splitting data, training a Random Forest model and evaluation
- Jupyter Notebook for the notebook version

## 4. Files
- `Saifullah_StudentPerformance.ipynb` — notebook code and explanations
- `Saifullah_StudentPerformance.py` — Python script version
- `student_performance_dataset.csv` — synthetic demonstration dataset
- `requirements.txt` — Python dependencies
- `Saifullah_StudentPerformance_ProjectReport.docx` — project report
- `README.md` — this guide

## 5. Setup and run
Use Python 3.10 or later.

```bash
python -m venv .venv
```

Activate the environment:
- Windows: `.venv\Scripts\activate`
- macOS/Linux: `source .venv/bin/activate`

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the script:

```bash
python Saifullah_StudentPerformance.py
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `Saifullah_StudentPerformance.ipynb` and run the cells from top to bottom.

## 6. Outputs
The script creates an `outputs` folder containing:
- `performance_categories.png`
- `study_hours_vs_score.png`
- `confusion_matrix.png`
- `feature_importance.png`

## 7. Method
1. Load the included CSV file.
2. Inspect the first rows, summary statistics and category counts.
3. Plot simple relationships and category totals.
4. Split records into training and test sets.
5. Train a Random Forest classifier on five input features.
6. Evaluate it on the test set using accuracy, a classification report and a confusion matrix.
7. Print one hypothetical example prediction.

## 8. Limitations and responsible use
- The dataset is synthetic, so results demonstrate the workflow rather than real-world student outcomes.
- Accuracy on this generated dataset does not prove that the model will work on real data.
- Predictions may reflect patterns deliberately built into the synthetic data.
- Do not use this model to grade, rank, label or make decisions about real students.
- If real data is used in a future version, obtain permission, remove identifying details, document its source and check for bias.

## 9. Dataset link
The dataset used for this submission is the local file `student_performance_dataset.csv` included with this project. It was generated for this project; it does not have an external source URL.

## 10. Before submission
- Run both the `.py` file and notebook.
- Confirm all charts are created and no errors appear.
- Add your own name/college details to the report if required.
- Follow the exact naming and upload instructions shown in your internship dashboard.
