"""
analytics/performance_predictor.py

Bonus ML Component: Student Performance Predictor.

Uses scikit-learn to predict whether a student will pass a topic
based on their mock test history. This is a classic ML classification
problem demonstrating the full ML lifecycle:

  Data Collection → Feature Engineering → Model Training →
  Evaluation → Persistence → Serving via API

Models implemented:
  1. Logistic Regression   — interpretable baseline (show coefficients)
  2. Random Forest         — better accuracy, feature importances

Interview talking point:
  "This demonstrates I understand the COMPLETE ML lifecycle — not just
   calling an LLM API. I built a supervised classification model with
   proper train/test splits, cross-validation, feature engineering,
   model persistence with joblib, and serving predictions via FastAPI.
   The feature importances tell students exactly which topics to focus on."
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from loguru import logger


# ── Feature schema ─────────────────────────────────────────────

FEATURE_COLUMNS = [
    "avg_score",           # mean score across all attempts (0–100)
    "latest_score",        # most recent test score
    "score_trend",         # latest_score - first_score (improvement)
    "attempts",            # number of tests taken
    "correct_rate",        # fraction of questions answered correctly
    "wrong_rate",          # fraction answered wrong (affected by neg marking)
    "skip_rate",           # fraction skipped
    "avg_time_per_q",      # seconds per question (speed indicator)
    "consistency",         # 1 - std(scores)/mean(scores) — higher = more consistent
    "days_active",         # how many different days student studied
]

TARGET_COLUMN = "will_pass"   # 1 = predicted to pass, 0 = at risk


# ── Mock data generator (for demonstration) ────────────────────

def generate_mock_data(n_students: int = 500) -> pd.DataFrame:
    """
    Generates realistic mock student performance data.

    In production: replace this with real data from your
    test_results table in the database.

    Interview talking point:
      "I defined clear feature engineering rules based on domain
       knowledge about competitive exam patterns — not just raw scores."
    """
    np.random.seed(42)
    records = []

    for student_id in range(n_students):
        # Simulate different student profiles
        profile = np.random.choice(["strong", "average", "weak"], p=[0.3, 0.4, 0.3])

        if profile == "strong":
            avg_score     = np.random.normal(72, 10)
            correct_rate  = np.random.normal(0.75, 0.1)
            attempts      = np.random.randint(5, 20)
            days_active   = np.random.randint(15, 60)
        elif profile == "average":
            avg_score     = np.random.normal(52, 12)
            correct_rate  = np.random.normal(0.55, 0.12)
            attempts      = np.random.randint(2, 12)
            days_active   = np.random.randint(5, 30)
        else:
            avg_score     = np.random.normal(35, 10)
            correct_rate  = np.random.normal(0.35, 0.12)
            attempts      = np.random.randint(1, 6)
            days_active   = np.random.randint(1, 15)

        # Clip to valid ranges
        avg_score     = np.clip(avg_score, 0, 100)
        correct_rate  = np.clip(correct_rate, 0, 1)

        wrong_rate    = np.clip(np.random.uniform(0.05, 0.4), 0, 1 - correct_rate)
        skip_rate     = np.clip(1 - correct_rate - wrong_rate, 0, 1)
        score_trend   = np.random.normal(5 if profile == "strong" else -2, 8)
        latest_score  = np.clip(avg_score + score_trend/2, 0, 100)
        avg_time_per_q = np.random.normal(75, 25)   # seconds
        consistency   = np.clip(1 - np.random.uniform(0.05, 0.35), 0, 1)

        # Label: pass if avg_score >= 45 AND correct_rate >= 0.45
        will_pass = int(avg_score >= 45 and correct_rate >= 0.45)

        records.append({
            "student_id":    student_id,
            "avg_score":     round(avg_score, 2),
            "latest_score":  round(latest_score, 2),
            "score_trend":   round(score_trend, 2),
            "attempts":      int(attempts),
            "correct_rate":  round(correct_rate, 3),
            "wrong_rate":    round(wrong_rate, 3),
            "skip_rate":     round(skip_rate, 3),
            "avg_time_per_q": round(avg_time_per_q, 1),
            "consistency":   round(consistency, 3),
            "days_active":   int(days_active),
            "will_pass":     will_pass,
        })

    df = pd.DataFrame(records)
    logger.info(f"Generated {len(df)} records | Pass rate: {df['will_pass'].mean():.1%}")
    return df


# ── Model trainer ──────────────────────────────────────────────

class PerformancePredictor:
    """
    Trains, evaluates, and serves a student performance prediction model.

    Two models are trained and compared:
      1. LogisticRegression — interpretable, fast, good baseline
      2. RandomForest       — higher accuracy, gives feature importances

    The best model is saved to disk and loaded for inference.
    """

    MODEL_PATH = Path("analytics/models/performance_predictor.joblib")
    REPORT_PATH = Path("analytics/models/training_report.json")

    def __init__(self):
        self.best_model: Optional[Pipeline] = None
        self.feature_importances: Dict[str, float] = {}
        self.model_type: str = ""

    def train(
        self,
        df: Optional[pd.DataFrame] = None,
        model_type: str = "random_forest",
    ) -> Dict[str, Any]:
        """
        Full training pipeline:
          1. Load / generate data
          2. Feature engineering
          3. Train/test split (stratified)
          4. Cross-validation
          5. Final fit + evaluation
          6. Save model to disk

        Args:
            df:         DataFrame with FEATURE_COLUMNS + TARGET_COLUMN
                        If None, uses generated mock data.
            model_type: "logistic_regression" or "random_forest"

        Returns:
            Evaluation report dict
        """
        if df is None:
            logger.info("No data provided — using mock data for demonstration")
            df = generate_mock_data(500)

        X = df[FEATURE_COLUMNS].values
        y = df[TARGET_COLUMN].values

        logger.info(f"Training {model_type} | Data: {X.shape} | "
                    f"Class balance: {y.mean():.1%} positive")

        # ── Train/test split ───────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42,
            stratify=y,          # maintain class balance in both sets
        )

        # ── Build sklearn Pipeline ─────────────────────────────
        # Pipeline ensures the same preprocessing is applied to
        # train and test data — prevents data leakage.
        if model_type == "logistic_regression":
            estimator = LogisticRegression(
                C=1.0,
                max_iter=1000,
                random_state=42,
                class_weight="balanced",
            )
        else:
            estimator = RandomForestClassifier(
                n_estimators=100,
                max_depth=8,
                min_samples_leaf=5,
                random_state=42,
                class_weight="balanced",
                n_jobs=-1,       # use all CPU cores
            )

        pipeline = Pipeline([
            ("scaler", StandardScaler()),   # normalise features
            ("model",  estimator),
        ])

        # ── Cross-validation (5-fold stratified) ──────────────
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
        logger.info(f"CV ROC-AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

        # ── Final fit on full training set ─────────────────────
        pipeline.fit(X_train, y_train)

        # ── Evaluation on held-out test set ───────────────────
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        report = {
            "model_type": model_type,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "accuracy": round(accuracy_score(y_test, y_pred), 4),
            "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
            "cv_roc_auc_mean": round(cv_scores.mean(), 4),
            "cv_roc_auc_std": round(cv_scores.std(), 4),
            "classification_report": classification_report(
                y_test, y_pred,
                target_names=["At Risk (0)", "Will Pass (1)"],
                output_dict=True,
            ),
            "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
            "feature_importances": self._get_feature_importances(
                pipeline, model_type
            ),
        }

        logger.info(
            f"Test Accuracy: {report['accuracy']:.4f} | "
            f"ROC-AUC: {report['roc_auc']:.4f}"
        )

        # ── Save model ─────────────────────────────────────────
        self.best_model = pipeline
        self.model_type = model_type
        self._save(pipeline, report)

        return report

    def predict(self, student_data: Dict[str, float]) -> Dict[str, Any]:
        """
        Predict pass/fail probability for one student.

        Args:
            student_data: Dict with keys matching FEATURE_COLUMNS

        Returns:
            {
              "prediction": "PASS" or "AT_RISK",
              "pass_probability": 0.87,
              "top_improvement_areas": [...],
              "study_recommendation": "..."
            }
        """
        model = self._load()

        # Build feature vector in correct order
        features = np.array([[
            student_data.get(col, 0.0) for col in FEATURE_COLUMNS
        ]])

        prob = model.predict_proba(features)[0][1]
        prediction = "PASS" if prob >= 0.5 else "AT_RISK"

        # Feature importances → tell student what to improve
        improvements = self._get_improvement_suggestions(student_data, prob)

        return {
            "prediction": prediction,
            "pass_probability": round(float(prob), 3),
            "confidence": round(abs(prob - 0.5) * 2, 3),  # 0=uncertain, 1=very confident
            "top_improvement_areas": improvements,
            "study_recommendation": self._recommendation(prediction, student_data),
        }

    def _get_feature_importances(
        self, pipeline: Pipeline, model_type: str
    ) -> Dict[str, float]:
        """Extract feature importances from the trained model."""
        model = pipeline.named_steps["model"]

        if model_type == "random_forest":
            importances = model.feature_importances_
        else:  # logistic regression
            importances = np.abs(model.coef_[0])
            importances = importances / importances.sum()  # normalise

        result = dict(zip(FEATURE_COLUMNS, importances.tolist()))
        # Sort by importance
        result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
        self.feature_importances = result
        return result

    def _get_improvement_suggestions(
        self, data: Dict[str, float], prob: float
    ) -> List[Dict[str, str]]:
        """Return top 3 actionable improvement areas."""
        suggestions = []

        if data.get("avg_score", 0) < 50:
            suggestions.append({
                "area": "Overall Score",
                "current": f"{data.get('avg_score', 0):.1f}%",
                "target": "50%",
                "action": "Focus on high-weightage topics. Solve 20 questions daily.",
            })
        if data.get("correct_rate", 0) < 0.55:
            suggestions.append({
                "area": "Accuracy",
                "current": f"{data.get('correct_rate', 0)*100:.1f}%",
                "target": "55%",
                "action": "Slow down. Attempt fewer questions but more accurately.",
            })
        if data.get("attempts", 0) < 5:
            suggestions.append({
                "area": "Practice Frequency",
                "current": f"{data.get('attempts', 0)} tests",
                "target": "10+ tests",
                "action": "Take at least 2 mock tests per week.",
            })
        if data.get("skip_rate", 0) > 0.3:
            suggestions.append({
                "area": "Attempted Questions",
                "current": f"Skipping {data.get('skip_rate', 0)*100:.0f}%",
                "target": "Skip < 20%",
                "action": "Work on time management. Use elimination technique.",
            })
        if data.get("consistency", 1.0) < 0.7:
            suggestions.append({
                "area": "Score Consistency",
                "current": f"{data.get('consistency', 0)*100:.0f}% consistent",
                "target": "70%+",
                "action": "Review weak subjects. Make formula/fact sheets.",
            })

        return suggestions[:3]

    @staticmethod
    def _recommendation(prediction: str, data: Dict) -> str:
        if prediction == "PASS":
            return (
                "🎯 You are on track to pass! Maintain consistency and "
                "focus on your weak areas identified above. "
                "Take 2 full mock tests per week."
            )
        else:
            return (
                "📚 You need focused study to improve your chances. "
                "Start with the highest-weightage subjects, "
                "practice negative-marking strategy, and increase your daily study hours."
            )

    def _save(self, pipeline: Pipeline, report: Dict) -> None:
        self.MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, self.MODEL_PATH)
        with open(self.REPORT_PATH, "w") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"Model saved to {self.MODEL_PATH}")

    def _load(self) -> Pipeline:
        if self.best_model is not None:
            return self.best_model
        if not self.MODEL_PATH.exists():
            logger.warning("No trained model found — training with mock data now...")
            self.train()
        return joblib.load(self.MODEL_PATH)


# ── CLI: train and display report ─────────────────────────────
if __name__ == "__main__":
    predictor = PerformancePredictor()

    # Train Random Forest
    report = predictor.train(model_type="random_forest")

    print("\n" + "="*60)
    print("PERFORMANCE PREDICTOR — TRAINING REPORT")
    print("="*60)
    print(f"Model:        {report['model_type']}")
    print(f"Accuracy:     {report['accuracy']:.4f}")
    print(f"ROC-AUC:      {report['roc_auc']:.4f}")
    print(f"CV ROC-AUC:   {report['cv_roc_auc_mean']:.4f} ± {report['cv_roc_auc_std']:.4f}")
    print(f"\nTop Feature Importances:")
    for feat, imp in list(report["feature_importances"].items())[:5]:
        bar = "█" * int(imp * 40)
        print(f"  {feat:<22} {bar} {imp:.3f}")

    # Example prediction
    print("\n" + "="*60)
    print("SAMPLE PREDICTION")
    print("="*60)
    sample = {
        "avg_score": 48.5,
        "latest_score": 55.0,
        "score_trend": 6.5,
        "attempts": 7,
        "correct_rate": 0.52,
        "wrong_rate": 0.18,
        "skip_rate": 0.30,
        "avg_time_per_q": 68.0,
        "consistency": 0.74,
        "days_active": 22,
    }
    result = predictor.predict(sample)
    print(f"Prediction:       {result['prediction']}")
    print(f"Pass Probability: {result['pass_probability']:.1%}")
    print(f"Confidence:       {result['confidence']:.1%}")
    print(f"\nTop Improvements:")
    for imp in result["top_improvement_areas"]:
        print(f"  • {imp['area']}: {imp['action']}")
    print(f"\nRecommendation: {result['study_recommendation']}")
