import glob
import logging
import math
import os
import sys
import joblib
import numpy as np
import pandas as pd
import sklearn
from flask import Flask, jsonify, request
from flask_cors import CORS
from werkzeug.exceptions import HTTPException

app = Flask(__name__)
app.json.sort_keys = False
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

_DEFAULT_ORIGINS = [
    "http://localhost:5000", "http://127.0.0.1:5000",
    "http://localhost:5500", "http://127.0.0.1:5500",
    "https://localhost:443", "https://127.0.0.1:443",
    "http://localhost:63342",
    "null",
]
_EXTRA_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOWED_ORIGINS", "").split(",") if o.strip()]
CORS(app, resources={r"/api/*": {"origins": _DEFAULT_ORIGINS + _EXTRA_ORIGINS}})

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.getenv("MODEL_DIR", os.path.join(BASE_DIR, "model"))


def _artifact_path(filename):
    exact = os.path.join(MODEL_DIR, filename)
    if os.path.exists(exact):
        return exact
    stamped = sorted(glob.glob(os.path.join(MODEL_DIR, f"*_{filename}")))
    if stamped:
        return stamped[-1]
    raise FileNotFoundError(f"Model artifact '{filename}' not found in {MODEL_DIR}")


regression_model = joblib.load(_artifact_path("rent_price_model.joblib"))
classification_pipeline = joblib.load(
    _artifact_path("logisticregression_classifier_for_employee_attrition.joblib"))
clustering_artifact = joblib.load(_artifact_path("kmeans_retail_store_segmentation.joblib"))
arm_artifact = joblib.load(_artifact_path("apriori_music_session_rules.joblib"))

clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]


def _find_column_transformer(estimator):
    if hasattr(estimator, "regressor_"):
        estimator = estimator.regressor_
    for _, step in getattr(estimator, "steps", []):
        if hasattr(step, "transformers_"):
            return step
    raise ValueError("No fitted ColumnTransformer found in the pipeline")


def _category_levels(estimator):
    """{column: [allowed levels]} for every OneHotEncoder / OrdinalEncoder in the pipeline."""
    levels = {}
    for _, transformer, columns in _find_column_transformer(estimator).transformers_:
        for _, step in getattr(transformer, "steps", []):
            if hasattr(step, "categories_"):
                for column, cats in zip(columns, step.categories_):
                    levels[column] = [str(c) for c in cats]
    return levels


RENT_LEVELS = _category_levels(regression_model)
ATTRITION_LEVELS = _category_levels(classification_pipeline)
RENT_COLUMNS = list(_find_column_transformer(regression_model).feature_names_in_)
ATTRITION_COLUMNS = list(_find_column_transformer(classification_pipeline).feature_names_in_)
STORE_COLUMNS = list(clustering_scaler.feature_names_in_)

_rent_estimator = regression_model.regressor_.named_steps["model"]
RENT_MODEL_LABEL = (f"{type(_rent_estimator).__name__}"
                    f"{'(alpha=' + str(_rent_estimator.alpha) + ')' if hasattr(_rent_estimator, 'alpha') else ''}"
                    " trained on log1p(monthly_rent_kes)")
_clf_estimator = classification_pipeline.named_steps["model"]
ATTRITION_CLASSES = [str(c) for c in _clf_estimator.classes_]
ATTRITION_MODEL_LABEL = (f"{type(_clf_estimator).__name__}(C={_clf_estimator.C}) "
                         "(SMOTE-resampled at training time only)")

SEGMENT_PROFILES = {
    0: {"segment_name": "Small neighbourhood stores",
        "profile": "Low foot traffic (~180/day), ~4 staff, ~110 sqm, ~KES 0.48M monthly revenue, small baskets (~KES 90)."},
    1: {"segment_name": "Flagship stores",
        "profile": "Very large (~850 sqm, ~22 staff), ~900 visitors/day, ~KES 4.7M monthly revenue, baskets ~KES 180."},
    2: {"segment_name": "High-traffic, low-basket convenience stores",
        "profile": "~690 visitors/day but the smallest baskets (~KES 50); ~8 staff, ~230 sqm, ~KES 0.64M revenue."},
    3: {"segment_name": "Low-traffic, high-basket specialty stores",
        "profile": "Few visitors (~145/day) but the biggest baskets (~KES 260); ~5 staff, ~100 sqm, ~KES 1.08M revenue."},
    4: {"segment_name": "Mid-size core stores",
        "profile": "Balanced mid-market stores: ~10 staff, ~320 sqm, ~400 visitors/day, baskets ~KES 120, ~KES 1.39M revenue."},
}

log.info("All four artifacts loaded from %s. The API is ready.", MODEL_DIR)


class ValidationError(Exception):
    def __init__(self, errors):
        super().__init__("Validation failed")
        self.errors = errors


def _num(minimum=None, maximum=None, **kw):
    return {"kind": "number", "min": minimum, "max": maximum, **kw}


def _int(minimum=None, maximum=None, **kw):
    return {"kind": "integer", "min": minimum, "max": maximum, **kw}


def _cat(levels, **kw):
    return {"kind": "category", "levels": levels, **kw}


def _coerce(spec, value):
    """Convert one JSON value to the type the model expects, or raise ValueError(message)."""
    if spec["kind"] == "category":
        if not isinstance(value, str):
            raise ValueError(f"must be a string, one of {spec['levels']}")
        canonical = {level.lower(): level for level in spec["levels"]}
        key = value.strip().lower()
        if key not in canonical:
            raise ValueError(f"'{value}' is not allowed; must be one of {spec['levels']}")
        return canonical[key]

    if isinstance(value, bool):
        raise ValueError("must be a number, not true/false")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError("must be a number")
    if not math.isfinite(number):
        raise ValueError("must be a finite number")
    if spec["kind"] == "integer":
        if number != int(number):
            raise ValueError("must be a whole number")
        number = int(number)
    if spec["min"] is not None and number < spec["min"]:
        raise ValueError(f"must be >= {spec['min']}")
    if spec["max"] is not None and number > spec["max"]:
        raise ValueError(f"must be <= {spec['max']}")
    return float(number)


def build_row(payload, fields):
    if not isinstance(payload, dict):
        raise ValidationError({"_body": "The JSON body must be an object of named fields."})
    errors, row = {}, {}
    for name, spec in fields.items():
        value = payload.get(name)
        if value is None:
            if spec.get("required"):
                errors[name] = "required"
            elif spec.get("nullable"):
                row[name] = float("nan")
            else:
                row[name] = spec.get("default")
            continue
        try:
            row[name] = _coerce(spec, value)
        except ValueError as exc:
            errors[name] = str(exc)
    if errors:
        raise ValidationError(errors)
    return row, sorted(set(payload) - set(fields))


def _json_body():
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError({"_body": "Send a JSON body with Content-Type: application/json."})
    return payload


RENT_FIELDS = {
    "property_type": _cat(RENT_LEVELS["property_type"], required=True),
    "neighborhood": _cat(RENT_LEVELS["neighborhood"], required=True),
    "furnishing_status": _cat(RENT_LEVELS["furnishing_status"], required=True),
    "size_sqm": _num(10, 1000, required=True),
    "num_bedrooms": _int(0, 12, required=True),
    "num_bathrooms": _int(1, 12, required=True),
    "distance_to_cbd_km": _num(0, 200, required=True),
    "service_charge_kes": _num(0, 1_000_000, required=True),
    "garage_size_sqm": _num(0, 200, default=0.0),
    "has_garage": _int(0, 1, nullable=True),
    "security_rating": _num(1, 5, nullable=True),
    "amenities_score": _num(0, 100, required=True),
    "nearby_schools_count": _int(0, 50, required=True),
}

ATTRITION_FIELDS = {
    "department": _cat(ATTRITION_LEVELS["department"], required=True),
    "job_role_level": _cat(ATTRITION_LEVELS["job_role_level"], required=True),
    "education_level": _cat(ATTRITION_LEVELS["education_level"], required=True),
    "performance_rating": _cat(ATTRITION_LEVELS["performance_rating"], required=True),
    "years_since_last_promotion": _num(0, 60, required=True),
    "monthly_salary_kes": _num(1, 5_000_000, required=True),
    "overtime_hours_per_month": _num(0, 300, required=True),
    "job_satisfaction_score": _num(1, 5, required=True),
    "work_life_balance_score": _num(1, 5, nullable=True),
    "distance_from_home_km": _num(0, 500, required=True),
    "num_companies_worked_previously": _int(0, 30, required=True),
    "remote_work_days_per_week": _int(0, 7, required=True),
    "bonus_amount_kes": _num(0, 10_000_000, nullable=True),
}

STORE_FIELDS = {
    "monthly_revenue_kes": _num(0, 100_000_000, required=True),
    "daily_foot_traffic": _num(0, 100_000, required=True),
    "staff_count": _num(1, 1_000, required=True),
    "store_size_sqm": _num(1, 100_000, required=True),
    "avg_basket_size_kes": _num(0, 1_000_000, nullable=True),
}

GARAGE_PROPERTY_TYPES = {"Bungalow", "Maisonette", "Townhouse"}


def _with_ignored(response, ignored):
    if ignored:
        response["ignored_fields"] = ignored
    return response


# =============================================================================
# 1. REGRESSION - predict monthly rent (KES) for a Nairobi rental listing
# =============================================================================
# curl -X POST http://127.0.0.1:5000/api/v1/models/rent-price-regressor/predictions \
#   -H "Content-Type: application/json" \
#   -d '{"property_type": "Apartment", "neighborhood": "Kilimani",
#        "furnishing_status": "Semi-Furnished", "size_sqm": 75, "num_bedrooms": 2,
#        "num_bathrooms": 2, "distance_to_cbd_km": 7.4, "service_charge_kes": 8000,
#        "garage_size_sqm": 0, "security_rating": null, "amenities_score": 60,
#        "nearby_schools_count": 2}'
#
# Notes
# * The saved model is a TransformedTargetRegressor: it was trained on log1p(rent) and
#   inverts that transform ITSELF inside .predict(). The response is therefore already in
#   KES - applying np.expm1 again here would be a bug (it would return ~e^rent).
# * security_rating = null is fine: the pipeline's own imputer fills it.
# * has_garage is an engineered feature from the notebook ("a garage size was recorded").
#   In the training data that is fully determined by property type (Bungalow, Maisonette and
#   Townhouse always have one; Apartment and Studio never do - see the notebook's crosstab).
#   Clients may omit it; we then derive it the same way, or from garage_size_sqm > 0.
@app.route("/api/v1/models/rent-price-regressor/predictions", methods=["POST"])
def predict_rent():
    row, ignored = build_row(_json_body(), RENT_FIELDS)
    warnings = []
    if math.isnan(row["has_garage"]):
        row["has_garage"] = 1.0 if (row["property_type"] in GARAGE_PROPERTY_TYPES
                                    or row["garage_size_sqm"] > 0) else 0.0
    if row["property_type"] in ("Apartment", "Studio") and row["has_garage"] == 1.0:
        warnings.append("No Apartment or Studio in the training data had a garage; treat this estimate with caution.")

    input_row = pd.DataFrame([row], columns=RENT_COLUMNS)
    predicted_rent_kes = float(regression_model.predict(input_row)[0])  # already inverse-transformed (KES)

    return jsonify(_with_ignored({
        "predicted_monthly_rent_kes": round(predicted_rent_kes, 2),
        "currency": "KES",
        "model": RENT_MODEL_LABEL,
        "warnings": warnings,
    }, ignored))


# =============================================================================
# 2. CLASSIFICATION - predict whether an employee is likely to leave (attrition)
# =============================================================================
# curl -X POST http://127.0.0.1:5000/api/v1/models/employee-attrition-classifier/predictions \
#   -H "Content-Type: application/json" \
#   -d '{"department": "Sales", "job_role_level": "Mid", "education_level": "Bachelor'"'"'s",
#        "performance_rating": "Average", "years_since_last_promotion": 6.5,
#        "monthly_salary_kes": 68000, "overtime_hours_per_month": 34,
#        "job_satisfaction_score": 1.8, "work_life_balance_score": 1.9,
#        "distance_from_home_km": 22, "num_companies_worked_previously": 3,
#        "remote_work_days_per_week": 0, "bonus_amount_kes": 8500}'
#
# Notes
# * bonus_amount_kes = null is meaningful: Operations has no bonus scheme (structural
#   missingness). The pipeline's imputer fills 0 and sets a missingness indicator itself.
# * The SMOTE step in the pipeline only runs during fit(); predict() skips resamplers.
@app.route("/api/v1/models/employee-attrition-classifier/predictions", methods=["POST"])
def predict_attrition():
    row, ignored = build_row(_json_body(), ATTRITION_FIELDS)
    input_row = pd.DataFrame([row], columns=ATTRITION_COLUMNS)

    predicted_class = str(classification_pipeline.predict(input_row)[0])
    probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_probabilities = {label: round(float(p), 4) for label, p in zip(ATTRITION_CLASSES, probabilities)}

    return jsonify(_with_ignored({
        "predicted_attrition": predicted_class,
        "attrition_probability": class_probabilities.get("Yes"),
        "class_probabilities": class_probabilities,
        "model": ATTRITION_MODEL_LABEL,
    }, ignored))


# =============================================================================
# 3. CLUSTERING - assign a (new) retail store to one of the five store segments
# =============================================================================
# curl -X POST http://127.0.0.1:5000/api/v1/models/retail-store-segmenter/predictions \
#   -H "Content-Type: application/json" \
#   -d '{"monthly_revenue_kes": 620000, "daily_foot_traffic": 640, "staff_count": 7,
#        "store_size_sqm": 210, "avg_basket_size_kes": 48}'
#
# Unlike the two Pipelines above, K-Means does not bundle its preprocessing, so the three
# objects the clustering notebook persisted together are applied in the same order as in
# training: imputer.transform -> scaler.transform -> kmeans.predict.
@app.route("/api/v1/models/retail-store-segmenter/predictions", methods=["POST"])
def predict_store_segment():
    row, ignored = build_row(_json_body(), STORE_FIELDS)
    input_row = pd.DataFrame([row], columns=STORE_COLUMNS)
    imputed_fields = [c for c in STORE_COLUMNS if pd.isna(input_row.at[0, c])]

    imputed = pd.DataFrame(clustering_imputer.transform(input_row), columns=STORE_COLUMNS)
    scaled = clustering_scaler.transform(imputed)
    cluster = int(clustering_kmeans.predict(scaled)[0])
    profile = SEGMENT_PROFILES.get(cluster, {})

    return jsonify(_with_ignored({
        "assigned_cluster": cluster,
        "segment_name": profile.get("segment_name"),
        "segment_profile": profile.get("profile"),
        "imputed_fields": imputed_fields,
        "model": f"K-Means (k={clustering_kmeans.n_clusters}, median-imputed + standardised features)",
    }, ignored))


# =============================================================================
# 4. ASSOCIATION RULE MINING - "Play Next" recommendations for a listening session
# =============================================================================
# curl -X POST http://127.0.0.1:5000/api/v1/models/music-recommender/predictions \
#   -H "Content-Type: application/json" \
#   -d '{"session_tracks": ["M08", "M10"], "top_n": 3}'
#
# There is no scikit-learn .predict() here. The mined rules were pre-computed into a
# lookup (antecedent track-set -> ranked consequents) in the notebook's persistence step.
# This endpoint only does lookup + rank, exactly as a production recommender service
# would; it never mines the rules again. ("cart" is accepted as an alias of
# "session_tracks" so the course's e-commerce examples still work.)
@app.route("/api/v1/models/music-recommender/predictions", methods=["POST"])
def recommend_tracks():
    payload = _json_body()
    if not isinstance(payload, dict):
        raise ValidationError({"_body": "The JSON body must be an object of named fields."})

    tracks = payload.get("session_tracks", payload.get("cart"))
    errors = {}
    if not isinstance(tracks, list) or not tracks:
        errors["session_tracks"] = "required: a non-empty list of track IDs, e.g. [\"M08\", \"M10\"]"
    else:
        known = set(track_lookup.index)
        tracks = [str(t).strip().upper() for t in tracks]
        unknown = sorted(set(tracks) - known)
        if unknown:
            errors["session_tracks"] = f"unknown track IDs {unknown}; valid IDs are {sorted(known)}"
    top_n = 3
    try:
        if payload.get("top_n") is not None:
            top_n = int(_coerce(_int(1, 10), payload["top_n"]))
    except ValueError as exc:
        errors["top_n"] = str(exc)
    if errors:
        raise ValidationError(errors)

    session = set(tracks)
    candidates = {}
    for antecedent, consequents in recommendation_lookup.items():
        if antecedent.issubset(session):  # every track of the rule is already in the session
            for track, confidence, lift in consequents:
                if track not in session and (track not in candidates or confidence > candidates[track][0]):
                    candidates[track] = (confidence, lift)
    ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:top_n]

    recommendations = [{
        "track_id": str(track),
        "track_name": str(track_lookup.loc[track, "item_name"]),
        "category": str(track_lookup.loc[track, "category"]),
        "confidence": round(float(confidence), 3),
        "lift": round(float(lift), 2),
    } for track, (confidence, lift) in ranked]

    response = {
        "session_tracks": sorted(session),
        "session_track_names": [str(track_lookup.loc[t, "item_name"]) for t in sorted(session)],
        "recommendations": recommendations,
        "model": "Apriori association rules (cleaned, pre-computed lookup)",
    }
    if not recommendations:
        response["message"] = ("No mined rule matches this session. Rules exist for combinations of "
                               "Gospel, Fitness/EDM, Instrumental, Hip-Hop/Amapiano and R&B tracks.")
    return jsonify(response)


ENDPOINTS = {
    "rent-price-regressor": "Regression: Nairobi monthly rent (KES)",
    "employee-attrition-classifier": "Classification: employee attrition (No/Yes)",
    "retail-store-segmenter": "Clustering: retail store segment (0-4)",
    "music-recommender": "Association rules: 'Play Next' track recommendations",
}


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "BBT 4206 - Serving ML models through a RESTish API",
        "status": "ok",
        "endpoints (POST, JSON)": {f"/api/v1/models/{slug}/predictions": desc for slug, desc in ENDPOINTS.items()},
        "catalogue (GET)": "/api/v1/models",
        "health (GET)": "/api/v1/health",
    })


@app.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models_loaded": ["rent_price_model", "employee_attrition_classifier",
                          "retail_store_kmeans", "music_apriori_rules"],
        "versions": {"python": sys.version.split()[0], "scikit-learn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    })


@app.route("/api/v1/models", methods=["GET"])
def catalogue():
    def describe(fields):
        return {name: ({"type": "category", "allowed": spec["levels"]} if spec["kind"] == "category"
                       else {"type": spec["kind"], "min": spec["min"], "max": spec["max"]})
                      | ({"required": True} if spec.get("required") else {"optional": True})
                for name, spec in fields.items()}

    return jsonify({
        "rent-price-regressor": {"description": ENDPOINTS["rent-price-regressor"], "fields": describe(RENT_FIELDS)},
        "employee-attrition-classifier": {"description": ENDPOINTS["employee-attrition-classifier"],
                                          "fields": describe(ATTRITION_FIELDS)},
        "retail-store-segmenter": {"description": ENDPOINTS["retail-store-segmenter"], "fields": describe(STORE_FIELDS),
                                   "segments": {str(k): v["segment_name"] for k, v in SEGMENT_PROFILES.items()}},
        "music-recommender": {"description": ENDPOINTS["music-recommender"],
                              "fields": {"session_tracks": {"type": "list of track IDs", "required": True,
                                                            "allowed": sorted(track_lookup.index)},
                                         "top_n": {"type": "integer", "min": 1, "max": 10, "default": 3}},
                              "tracks": {t: str(n) for t, n in track_lookup["item_name"].sort_index().items()}},
    })


@app.errorhandler(ValidationError)
def handle_validation_error(err):
    return jsonify({"error": "Invalid input", "details": err.errors}), 422


@app.errorhandler(HTTPException)
def handle_http_error(err):
    return jsonify({"error": err.name, "details": err.description}), err.code


@app.errorhandler(Exception)
def handle_unexpected_error(err):
    log.exception("Unhandled error while serving %s", request.path)
    return jsonify({"error": "Internal Server Error",
                    "details": "The model could not produce a prediction for this input."}), 500


if __name__ == "__main__":
    # debug=True gives auto-reload and an interactive debugger - handy in class, but it can
    # expose a remote-code-execution console, so it is OFF unless FLASK_DEBUG=1 is set.
    app.run(host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "5000")),
            debug=os.getenv("FLASK_DEBUG") == "1")

    # HTTPS/TLS alternative for local development (self-signed cert; see
    # app_server_reverse_proxy_server_setup.md):
    # app.run(debug=False, ssl_context=("cert.pem", "key.pem"))
