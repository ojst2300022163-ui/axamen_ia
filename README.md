# Cardio AI - Sistema Inteligente para Detección de Enfermedades Cardíacas

Aplicación web en Flask con autenticación, predicción con ML (Logistic Regression y Random Forest), registro de predicciones en SQLite, visualización de métricas, y reentrenamiento con nuevos casos. Usa el dataset `heart.csv` incluido en el repositorio.

## Requisitos
- Python 3.10+

## Instalación y ejecución rápida

```bash
# 1) Moverse a la carpeta de trabajo
cd /workspace

# 2) Crear y activar un entorno virtual (opcional pero recomendado)
python3 -m venv .venv
source .venv/bin/activate

# 3) Instalar dependencias
pip install --upgrade pip
pip install -r requirements.txt

# 4) Ejecutar la aplicación
python run.py
```

La app quedará disponible en `http://localhost:5000`.

- Usuario inicial: `admin`
- Contraseña inicial: `admin123` (puedes cambiarla con la variable de entorno `ADMIN_PASSWORD` antes de iniciar)

## Variables de entorno (opcional)
- `SECRET_KEY`: clave para sesiones Flask (por defecto `dev-secret-key`).
- `DATABASE_URL`: URL de la base de datos (por defecto `sqlite:///app.db`).
- `ADMIN_PASSWORD`: contraseña del usuario `admin` inicial (por defecto `admin123`).

Ejemplo:
```bash
export ADMIN_PASSWORD="cambia-esto"
python run.py
```

## Funcionalidades
- Login y registro de usuarios (página `/login` y `/register`).
- Formulario de predicción (`/predict`) con selección de modelo.
- Registro de cada predicción con usuario, caso y probabilidad.
- Métricas y matrices de confusión en `/metrics`.
- Agregar casos con etiqueta real en `/add_case`.
- Reentrenar modelos con todos los datos disponibles en `/retrain`.
- Gestión básica de usuarios (solo `admin`) en `/users`.

## Estructura de datos
El dataset `heart.csv` contiene las siguientes columnas de entrada y la etiqueta objetivo:

Entrada (`FEATURE_COLUMNS`):
- age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal

Etiqueta (`target`):
- 0 = No enfermedad
- 1 = Sí enfermedad

## Cómo reentrenar
1. Agrega uno o más casos confirmados en `/add_case` (incluye `target`).
2. Ve a `/retrain` y presiona “Reentrenar con todos los datos”.
3. Revisa `/metrics` para ver las nuevas métricas y matrices de confusión.

## Notas
- Los modelos y métricas se guardan en `./models` y las imágenes en `./static/metrics`.
- La base de datos SQLite se crea en `app.db` en el directorio raíz.