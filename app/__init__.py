import os
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Globals initialized later
db = SQLAlchemy()
login_manager = LoginManager()


def create_app():
    # Serve static files from /workspace/static so saved metrics images are accessible
    app = Flask(__name__, static_folder=os.path.join("/workspace", "static"))
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get("DATABASE_URL", "sqlite:///app.db")
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = "login"

    # Local imports after db is set
    from .models import User, PatientCase, PredictionLog
    from .ml import (
        FEATURE_COLUMNS,
        ensure_models_trained,
        predict_with_model,
        train_and_persist_models,
        load_cached_metrics,
        load_base_dataset,
    )

    with app.app_context():
        db.create_all()
        # Seed default admin user if none exists
        if User.query.count() == 0:
            admin_password = os.environ.get("ADMIN_PASSWORD", "admin123")
            admin_user = User(username="admin")
            admin_user.set_password(admin_password)
            db.session.add(admin_user)
            db.session.commit()
            app.logger.info("Seeded default admin user 'admin'")

        # Train models on first run if missing
        ensure_models_trained()

    @login_manager.user_loader
    def load_user(user_id):
        return db.session.get(User, int(user_id))

    @app.route("/login", methods=["GET", "POST"])
    def login():
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            user = User.query.filter_by(username=username).first()
            if user and user.check_password(password):
                login_user(user)
                return redirect(url_for("predict"))
            flash("Credenciales inválidas", "danger")
        return render_template("login.html")

    @app.route("/register", methods=["GET", "POST"])
    def register():
        # Allow open registration so that new users can be created
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if not username or not password:
                flash("Usuario y contraseña son requeridos", "warning")
                return render_template("register.html")
            if User.query.filter_by(username=username).first():
                flash("El usuario ya existe", "warning")
                return render_template("register.html")
            user = User(username=username)
            user.set_password(password)
            db.session.add(user)
            db.session.commit()
            flash("Usuario creado. Ya puedes iniciar sesión.", "success")
            return redirect(url_for("login"))
        return render_template("register.html")

    @app.route("/logout")
    @login_required
    def logout():
        logout_user()
        return redirect(url_for("login"))

    @app.route("/")
    @login_required
    def home():
        return redirect(url_for("predict"))

    @app.route("/predict", methods=["GET", "POST"])
    @login_required
    def predict():
        result = None
        probability = None
        selected_model = request.form.get("model", "logreg")

        if request.method == "POST":
            input_data = {}
            errors = []
            for feature in FEATURE_COLUMNS:
                value = request.form.get(feature)
                if value is None or value == "":
                    errors.append(f"Falta el campo: {feature}")
                    continue
                try:
                    # All features are numeric
                    input_data[feature] = float(value)
                except ValueError:
                    errors.append(f"Valor inválido para {feature}")
            if errors:
                for e in errors:
                    flash(e, "danger")
            else:
                # Create a patient case without label (unknown)
                patient_case = PatientCase(**{k: input_data[k] for k in FEATURE_COLUMNS})
                db.session.add(patient_case)
                db.session.commit()

                pred_label, pred_proba = predict_with_model(selected_model, input_data)

                # Log prediction
                log = PredictionLog(
                    user_id=current_user.id,
                    patient_case_id=patient_case.id,
                    model_name=selected_model,
                    predicted_label=int(pred_label),
                    probability=float(pred_proba),
                )
                db.session.add(log)
                db.session.commit()

                result = "Sí" if int(pred_label) == 1 else "No"
                probability = round(float(pred_proba) * 100.0, 2)

        return render_template(
            "predict.html",
            feature_columns=FEATURE_COLUMNS,
            selected_model=selected_model,
            result=result,
            probability=probability,
        )

    @app.route("/metrics")
    @login_required
    def metrics():
        metrics_data = load_cached_metrics()
        return render_template("metrics.html", metrics=metrics_data)

    @app.route("/retrain", methods=["GET", "POST"])
    @login_required
    def retrain():
        if request.method == "POST":
            train_and_persist_models()
            flash("Modelos reentrenados correctamente.", "success")
            return redirect(url_for("metrics"))

        # Show counts
        base = load_base_dataset()
        base_count = int(base["X"].shape[0])
        from .models import PatientCase
        labeled_cases = PatientCase.query.filter(PatientCase.target.isnot(None)).count()
        return render_template(
            "retrain.html",
            base_count=base_count,
            extra_labeled_count=labeled_cases,
        )

    @app.route("/add_case", methods=["GET", "POST"])
    @login_required
    def add_case():
        from .ml import FEATURE_COLUMNS
        if request.method == "POST":
            input_data = {}
            errors = []
            for feature in FEATURE_COLUMNS:
                value = request.form.get(feature)
                if value is None or value == "":
                    errors.append(f"Falta el campo: {feature}")
                    continue
                try:
                    input_data[feature] = float(value)
                except ValueError:
                    errors.append(f"Valor inválido para {feature}")
            # target
            target_val = request.form.get("target")
            try:
                target_val = int(target_val)
                if target_val not in (0, 1):
                    raise ValueError()
            except Exception:
                errors.append("El campo target debe ser 0 o 1")

            if errors:
                for e in errors:
                    flash(e, "danger")
            else:
                from .models import PatientCase
                case = PatientCase(**input_data, target=target_val)
                db.session.add(case)
                db.session.commit()
                flash("Caso agregado con etiqueta real. Será usado en el reentrenamiento.", "success")
                return redirect(url_for("retrain"))

        return render_template("add_case.html", feature_columns=FEATURE_COLUMNS)

    @app.route("/users", methods=["GET", "POST"])
    @login_required
    def users():
        # Simple user creation form. Only allow admin user to create others.
        if current_user.username != "admin":
            flash("Solo el usuario admin puede gestionar usuarios.", "warning")
            return redirect(url_for("predict"))

        from .models import User
        if request.method == "POST":
            username = request.form.get("username", "").strip()
            password = request.form.get("password", "")
            if not username or not password:
                flash("Usuario y contraseña son requeridos", "warning")
            elif db.session.query(db.exists().where(User.username == username)).scalar():
                flash("El usuario ya existe", "warning")
            else:
                user = User(username=username)
                user.set_password(password)
                db.session.add(user)
                db.session.commit()
                flash("Usuario creado", "success")
                return redirect(url_for("users"))

        all_users = db.session.execute(db.select(User)).scalars().all()
        return render_template("users.html", users=all_users)

    return app