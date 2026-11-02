import glob
import os
import gradio as gr
import joblib
import pandas as pd

# Hugging Face's free Gradio hardware is ZeroGPU and some Spaces refuse to start unless one
# function is decorated with @spaces.GPU. Nothing here needs a GPU; this no-op only satisfies
# that platform check. Locally the `spaces` package does not exist, so we skip it.
try:
    import spaces

    @spaces.GPU
    def _warm_zerogpu():
        return None

    _warm_zerogpu()
except ImportError:
    pass

HERE = os.path.dirname(os.path.abspath(__file__))
SEARCH_DIRS = [HERE, os.path.join(HERE, "model"), os.path.join(HERE, "..", "model"), os.getcwd(), os.path.join(os.getcwd(), "model")]


def find_artifact(filename):
    for d in SEARCH_DIRS:
        for pattern in (filename, f"*_{filename}"):
            hits = sorted(glob.glob(os.path.join(d, pattern)))
            if hits:
                return hits[-1]
    raise FileNotFoundError(f"{filename} not found in {SEARCH_DIRS}")


# Load all four artifacts ONCE at import time, exactly like api.py.
rent_model = joblib.load(find_artifact("rent_price_model.joblib"))
attrition_pipeline = joblib.load(find_artifact("logisticregression_classifier_for_employee_attrition.joblib"))
store_artifact = joblib.load(find_artifact("kmeans_retail_store_segmentation.joblib"))
arm_artifact = joblib.load(find_artifact("apriori_music_session_rules.joblib"))
store_imputer, store_scaler, store_kmeans = store_artifact["imputer"], store_artifact["scaler"], store_artifact["kmeans"]
recommendation_lookup, track_lookup = arm_artifact["recommendation_lookup"], arm_artifact["track_lookup"]

SEGMENTS = {0: "Small neighbourhood stores", 1: "Flagship stores", 2: "High-traffic, low-basket convenience stores",
            3: "Low-traffic, high-basket specialty stores", 4: "Mid-size core stores"}
GARAGE_TYPES = {"Bungalow", "Maisonette", "Townhouse"}
NAN = float("nan")
TRACK_CHOICES = [(f"{r.item_name} ({t})", t) for t, r in track_lookup.sort_index().iterrows()]


def as_float_frame(row):
    """Cast every numeric column to float64, as api.py does. Gradio's Number returns a Python int for
    whole values (e.g. 8500), and the fitted imputers (trained on floats) reject int64 columns."""
    numeric = [c for c in row.columns if not pd.api.types.is_object_dtype(row[c]) and not pd.api.types.is_string_dtype(row[c])]
    return row.astype({c: "float64" for c in numeric})


def need(**fields):
    """Friendly error for empty Gradio inputs (cleared Number boxes arrive as None)."""
    missing = [k for k, v in fields.items() if v is None or v == ""]
    if missing:
        raise gr.Error("Please fill in: " + ", ".join(missing))


# 1. Regression ------------------------------------------------------------------------------------
def predict_rent(property_type, neighborhood, furnishing, size, beds, baths, dist, service, garage, security, amenities, schools):
    need(size=size, bedrooms=beds, bathrooms=baths, distance=dist, service_charge=service, amenities=amenities, schools=schools)
    garage = garage or 0
    row = pd.DataFrame([{
        "property_type": property_type, "neighborhood": neighborhood, "furnishing_status": furnishing,
        "size_sqm": size, "num_bedrooms": beds, "num_bathrooms": baths, "distance_to_cbd_km": dist,
        "service_charge_kes": service, "garage_size_sqm": garage,
        "security_rating": security if security and security >= 1 else NAN,
        "amenities_score": amenities, "nearby_schools_count": schools,
        "has_garage": 1.0 if (property_type in GARAGE_TYPES or garage > 0) else 0.0}])
    # TransformedTargetRegressor: predict() already returns KES (no expm1 needed).
    return f"KES {float(rent_model.predict(as_float_frame(row))[0]):,.0f} per month"


# 2. Classification --------------------------------------------------------------------------------
def predict_attrition(dept, level, edu, perf, salary, bonus, overtime, promo, sat, wlb, distance, prev, remote):
    need(salary=salary, overtime=overtime, years_since_promotion=promo, distance=distance, previous_employers=prev, remote_days=remote)
    row = pd.DataFrame([{
        "department": dept, "job_role_level": level, "education_level": edu, "performance_rating": perf,
        "years_since_last_promotion": promo, "monthly_salary_kes": salary, "overtime_hours_per_month": overtime,
        "job_satisfaction_score": sat, "work_life_balance_score": wlb, "distance_from_home_km": distance,
        "num_companies_worked_previously": prev, "remote_work_days_per_week": remote,
        "bonus_amount_kes": bonus if bonus and bonus > 0 else NAN}])
    row = as_float_frame(row)
    label = str(attrition_pipeline.predict(row)[0])
    probs = dict(zip(attrition_pipeline.named_steps["model"].classes_, attrition_pipeline.predict_proba(row)[0]))
    return (f"{'Likely to leave' if label == 'Yes' else 'Likely to stay'}  (prediction: {label})\n"
            f"Probability of leaving: {probs['Yes']:.1%}\nProbability of staying: {probs['No']:.1%}")


# 3. Clustering ------------------------------------------------------------------------------------
def predict_segment(revenue, traffic, staff, sqm, basket):
    need(revenue=revenue, visitors=traffic, staff=staff, size=sqm)
    cols = list(store_scaler.feature_names_in_)
    row = pd.DataFrame([[revenue, traffic, staff, sqm, basket if basket and basket > 0 else NAN]], columns=cols)
    row = as_float_frame(row)
    scaled = store_scaler.transform(pd.DataFrame(store_imputer.transform(row), columns=cols))
    cluster = int(store_kmeans.predict(scaled)[0])
    return f"Cluster {cluster}: {SEGMENTS[cluster]}"


# 4. Association rules -----------------------------------------------------------------------------
def recommend_tracks(played, top_n):
    session = set(played or [])
    if not session:
        raise gr.Error("Pick at least one track that has already played.")
    candidates = {}
    for antecedent, consequents in recommendation_lookup.items():
        if antecedent.issubset(session):
            for track, conf, lift in consequents:
                if track not in session and (track not in candidates or conf > candidates[track][0]):
                    candidates[track] = (conf, lift)
    ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[: int(top_n or 3)]
    if not ranked:
        return "No mined rule matches this session."
    return "\n".join(f"{i}. {track_lookup.loc[t, 'item_name']} ({t}): confidence {c:.1%}, lift {l:.2f}"
                     for i, (t, (c, l)) in enumerate(ranked, 1))


# One Blocks app, four tabs --------------------------------------------------------------------------
with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206: Model Validation Demos\nSame trained artifacts as the Flask API. Switch tabs to try each model.")

    with gr.Tab("Rent price (regression)"):
        gr.Interface(fn=predict_rent, flagging_mode="never", inputs=[
            gr.Dropdown(["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"], value="Apartment", label="Property type"),
            gr.Dropdown(["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"], value="Kilimani", label="Neighbourhood"),
            gr.Dropdown(["Unfurnished", "Semi-Furnished", "Fully Furnished"], value="Semi-Furnished", label="Furnishing"),
            gr.Number(75, label="Size (sqm)"), gr.Number(2, precision=0, label="Bedrooms"), gr.Number(2, precision=0, label="Bathrooms"),
            gr.Number(7.4, label="Distance to CBD (km)"), gr.Number(8000, label="Service charge (KES/month)"),
            gr.Number(0, label="Garage size (sqm, 0 = none)"), gr.Number(0, label="Security rating (1-5, 0 = unknown)"),
            gr.Number(60, label="Amenities score (0-100)"), gr.Number(2, precision=0, label="Schools nearby"),
        ], outputs=gr.Textbox(label="Predicted monthly rent"))

    with gr.Tab("Employee attrition (classification)"):
        gr.Interface(fn=predict_attrition, flagging_mode="never", inputs=[
            gr.Dropdown(["Engineering", "Finance", "HR", "Operations", "Sales"], value="Sales", label="Department"),
            gr.Dropdown(["Junior", "Mid", "Senior", "Lead"], value="Mid", label="Job level"),
            gr.Dropdown(["Diploma", "Bachelor's", "Master's", "PhD"], value="Bachelor's", label="Education"),
            gr.Dropdown(["Low", "Average", "High", "Outstanding"], value="Average", label="Performance rating"),
            gr.Number(68000, label="Monthly salary (KES)"), gr.Number(8500, label="Bonus (KES, 0 = no bonus scheme)"),
            gr.Number(34, label="Overtime (hours/month)"), gr.Number(6.5, label="Years since last promotion"),
            gr.Slider(1, 5, value=1.8, step=0.1, label="Job satisfaction (1-5)"),
            gr.Slider(1, 5, value=1.9, step=0.1, label="Work-life balance (1-5)"),
            gr.Number(22, label="Distance from home (km)"), gr.Number(3, precision=0, label="Previous employers"),
            gr.Number(0, precision=0, label="Remote days per week"),
        ], outputs=gr.Textbox(label="Prediction"))

    with gr.Tab("Store segment (clustering)"):
        gr.Interface(fn=predict_segment, flagging_mode="never", inputs=[
            gr.Number(620000, label="Monthly revenue (KES)"), gr.Number(640, label="Visitors per day"),
            gr.Number(7, precision=0, label="Staff"), gr.Number(210, label="Store size (sqm)"),
            gr.Number(48, label="Average basket (KES, 0 = unknown)"),
        ], outputs=gr.Textbox(label="Segment"))

    with gr.Tab("Play Next (association rules)"):
        gr.Interface(fn=recommend_tracks, flagging_mode="never", inputs=[
            gr.CheckboxGroup(TRACK_CHOICES, value=["M08", "M10"], label="Tracks already played"),
            gr.Number(3, precision=0, minimum=1, maximum=10, label="How many suggestions"),
        ], outputs=gr.Textbox(label="Recommendations"))

if __name__ == "__main__":
    demo.launch()
