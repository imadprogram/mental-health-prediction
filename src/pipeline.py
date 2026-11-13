import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

from src.preprocessing import build_preprocessor, load_and_split_data

# Dynamic absolute paths based on project root
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'student_mental_health.csv')
TRAIN_PATH = os.path.join(BASE_DIR, 'data', 'train.csv')
TEST_PATH = os.path.join(BASE_DIR, 'data', 'test.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'best_model.joblib')


def prepare_data():
    """Task 1: Load raw data, drop ID, map target, split and save to data/"""
    X_train, X_test, y_train, y_test = load_and_split_data(DATA_PATH)

    train_df = X_train.copy()
    train_df['Mental_Health_Risk'] = y_train
    train_df.to_csv(TRAIN_PATH, index=False)

    test_df = X_test.copy()
    test_df['Mental_Health_Risk'] = y_test
    test_df.to_csv(TEST_PATH, index=False)
    print("Data prepared and saved to train.csv and test.csv")


def train_model():
    """Task 2: Train winning Logistic Regression pipeline and save model"""
    train_df = pd.read_csv(TRAIN_PATH)
    X_train = train_df.drop('Mental_Health_Risk', axis=1)
    y_train = train_df['Mental_Health_Risk']

    preprocessor = build_preprocessor(X_train)
    model = LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=1000, random_state=42
    )

    pipe = Pipeline([('preprocessor', preprocessor), ('model', model)])

    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODEL_PATH)
    print(f"Model successfully trained and saved to {MODEL_PATH}")


def evaluate_model():
    """Task 3: Evaluate model on unseen test set and enforce quality gate"""
    test_df = pd.read_csv(TEST_PATH)
    X_test = test_df.drop('Mental_Health_Risk', axis=1)
    y_test = test_df['Mental_Health_Risk']

    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    auc = roc_auc_score(y_test, y_prob, multi_class='ovr')

    print(
        f"Evaluation Results -> Accuracy: {acc:.4f} | F1: {f1:.4f} | ROC-AUC:"
        f" {auc:.4f}"
    )

    # Quality Gate
    if acc < 0.70:
        raise ValueError(
            f"Quality Gate Failed: Model accuracy ({acc:.4f}) is below 0.70!"
        )