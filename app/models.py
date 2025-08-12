from datetime import datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from . import db


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class PatientCase(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.Float, nullable=False)
    sex = db.Column(db.Float, nullable=False)
    cp = db.Column(db.Float, nullable=False)
    trestbps = db.Column(db.Float, nullable=False)
    chol = db.Column(db.Float, nullable=False)
    fbs = db.Column(db.Float, nullable=False)
    restecg = db.Column(db.Float, nullable=False)
    thalach = db.Column(db.Float, nullable=False)
    exang = db.Column(db.Float, nullable=False)
    oldpeak = db.Column(db.Float, nullable=False)
    slope = db.Column(db.Float, nullable=False)
    ca = db.Column(db.Float, nullable=False)
    thal = db.Column(db.Float, nullable=False)
    target = db.Column(db.Integer, nullable=True)  # Real label if known
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


class PredictionLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    patient_case_id = db.Column(db.Integer, db.ForeignKey("patient_case.id"), nullable=False)
    model_name = db.Column(db.String(50), nullable=False)
    predicted_label = db.Column(db.Integer, nullable=False)
    probability = db.Column(db.Float, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)