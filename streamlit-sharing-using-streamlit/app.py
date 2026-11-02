import streamlit as st
import glob
import math
import os
import joblib
import pandas as pd


st.set_page_config(page_title="BBT 4206_a7 Model Validation ", page_icon="📊", layout="centered")

HERE = os.path.dirname(os.path.abspath(__file__))
CANDIDATE_DIRS = [os.path.join(os.getcwd(), "model"), os.path.join(HERE, "..", "model"), os.path.join(HERE, "model")]


def find_artifact(filename):
    for d in CANDIDATE_DIRS:
        for pattern in (filename, f"*_{filename}"):
            hits = sorted(glob.glob(os.path.join(d, pattern)))
            if hits:
                return hits[-1]
    raise FileNotFoundError(f"{filename} not found in any of {CANDIDATE_DIRS}")


@st.cache_resource
def load_artifacts():
    """Load once and reuse across reruns (Streamlit re-executes this script on every click)."""
    return (joblib.load(find_artifact("rent_price_model.joblib")),
            joblib.load(find_artifact("logisticregression_classifier_for_employee_attrition.joblib")),
            joblib.load(find_artifact("kmeans_retail_store_segmentation.joblib")),
            joblib.load(find_artifact("apriori_music_session_rules.joblib")))


rent_model, attrition_pipeline, store_artifact, arm_artifact = load_artifacts()
store_imputer, store_scaler, store_kmeans = store_artifact["imputer"], store_artifact["scaler"], store_artifact["kmeans"]
recommendation_lookup, track_lookup = arm_artifact["recommendation_lookup"], arm_artifact["track_lookup"]

SEGMENTS = {
    0: "Small neighbourhood stores", 1: "Flagship stores",
    2: "High-traffic, low-basket convenience stores",
    3: "Low-traffic, high-basket specialty stores", 4: "Mid-size core stores",
}
GARAGE_TYPES = {"Bungalow", "Maisonette", "Townhouse"}   # has_garage == 1 for these in the training data
NAN = float("nan")


def as_float_frame(row):
    """Cast numeric columns to float64, as api.py does (the fitted imputers reject int64 columns)."""
    numeric = [c for c in row.columns if not pd.api.types.is_object_dtype(row[c]) and not pd.api.types.is_string_dtype(row[c])]
    return row.astype({c: "float64" for c in numeric})

st.title("BBT 4206: Model Validation Demos")
st.caption("Same trained artifacts as the Flask API. Switch tabs to validate each model.")
tab_rent, tab_attr, tab_store, tab_music = st.tabs(
    ["Rent price (regression)", "Employee attrition (classification)",
     "Store segment (clustering)", "Play Next (association rules)"])

# ----------------------------------------------------------------------------- 1. Regression
with tab_rent:
    with st.form("rent_form"):
        c1, c2 = st.columns(2)
        property_type = c1.selectbox("Property type", ["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"])
        neighborhood = c2.selectbox("Neighbourhood", ["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"], index=1)
        furnishing = c1.selectbox("Furnishing", ["Unfurnished", "Semi-Furnished", "Fully Furnished"], index=1)
        size = c2.number_input("Size (sqm)", 10.0, 1000.0, 75.0)
        beds = c1.number_input("Bedrooms", 0, 12, 2)
        baths = c2.number_input("Bathrooms", 1, 12, 2)
        dist = c1.number_input("Distance to CBD (km)", 0.0, 200.0, 7.4)
        service = c2.number_input("Service charge (KES/month)", 0.0, 1_000_000.0, 8000.0)
        garage = c1.number_input("Garage size (sqm, 0 = none)", 0.0, 200.0, 0.0)
        security = c2.number_input("Security rating (1-5, 0 = unknown)", 0.0, 5.0, 0.0)
        amenities = c1.number_input("Amenities score (0-100)", 0.0, 100.0, 60.0)
        schools = c2.number_input("Schools nearby", 0, 50, 2)
        go = st.form_submit_button("Predict rent")
    if go:
        has_garage = 1.0 if (property_type in GARAGE_TYPES or garage > 0) else 0.0
        row = pd.DataFrame([{
            "property_type": property_type, "neighborhood": neighborhood, "furnishing_status": furnishing,
            "size_sqm": size, "num_bedrooms": beds, "num_bathrooms": baths, "distance_to_cbd_km": dist,
            "service_charge_kes": service, "garage_size_sqm": garage,
            "security_rating": security if security >= 1 else NAN,        # 0 -> missing -> imputed by the pipeline
            "amenities_score": amenities, "nearby_schools_count": schools, "has_garage": has_garage}])
        # The model is a TransformedTargetRegressor: predict() already returns KES (no expm1 needed).
        st.success(f"Predicted monthly rent: KES {float(rent_model.predict(as_float_frame(row))[0]):,.0f}")
        if property_type in ("Apartment", "Studio") and garage > 0:
            st.warning("No Apartment or Studio in the training data had a garage; treat this with caution.")

# ----------------------------------------------------------------------------- 2. Classification
with tab_attr:
    with st.form("attr_form"):
        c1, c2 = st.columns(2)
        dept = c1.selectbox("Department", ["Engineering", "Finance", "HR", "Operations", "Sales"], index=4)
        level = c2.selectbox("Job level", ["Junior", "Mid", "Senior", "Lead"], index=1)
        edu = c1.selectbox("Education", ["Diploma", "Bachelor's", "Master's", "PhD"], index=1)
        perf = c2.selectbox("Performance rating", ["Low", "Average", "High", "Outstanding"], index=1)
        salary = c1.number_input("Monthly salary (KES)", 1.0, 5_000_000.0, 68000.0)
        bonus = c2.number_input("Bonus (KES, 0 = no bonus scheme)", 0.0, 10_000_000.0, 8500.0)
        overtime = c1.number_input("Overtime (hours/month)", 0.0, 300.0, 34.0)
        promo = c2.number_input("Years since last promotion", 0.0, 60.0, 6.5)
        sat = c1.slider("Job satisfaction (1-5)", 1.0, 5.0, 1.8, 0.1)
        wlb = c2.slider("Work-life balance (1-5)", 1.0, 5.0, 1.9, 0.1)
        distance = c1.number_input("Distance from home (km)", 0.0, 500.0, 22.0)
        prev = c2.number_input("Previous employers", 0, 30, 3)
        remote = c1.number_input("Remote days per week", 0, 7, 0)
        go = st.form_submit_button("Predict attrition")
    if go:
        row = pd.DataFrame([{
            "department": dept, "job_role_level": level, "education_level": edu, "performance_rating": perf,
            "years_since_last_promotion": promo, "monthly_salary_kes": salary, "overtime_hours_per_month": overtime,
            "job_satisfaction_score": sat, "work_life_balance_score": wlb, "distance_from_home_km": distance,
            "num_companies_worked_previously": prev, "remote_work_days_per_week": remote,
            "bonus_amount_kes": bonus if bonus > 0 else NAN}])             # 0 -> no scheme -> pipeline fills it
        row = as_float_frame(row)
        label = str(attrition_pipeline.predict(row)[0])
        probs = dict(zip(attrition_pipeline.named_steps["model"].classes_, attrition_pipeline.predict_proba(row)[0]))
        (st.error if label == "Yes" else st.success)(
            f"{'Likely to leave' if label == 'Yes' else 'Likely to stay'} (probability of leaving {probs['Yes']:.1%})")
        st.write({k: round(float(v), 4) for k, v in probs.items()})

# ----------------------------------------------------------------------------- 3. Clustering
with tab_store:
    with st.form("store_form"):
        c1, c2 = st.columns(2)
        revenue = c1.number_input("Monthly revenue (KES)", 0.0, 100_000_000.0, 620000.0)
        traffic = c2.number_input("Visitors per day", 0.0, 100_000.0, 640.0)
        staff = c1.number_input("Staff", 1, 1000, 7)
        sqm = c2.number_input("Store size (sqm)", 1.0, 100_000.0, 210.0)
        basket = c1.number_input("Average basket (KES, 0 = unknown)", 0.0, 1_000_000.0, 48.0)
        go = st.form_submit_button("Find segment")
    if go:
        cols = list(store_scaler.feature_names_in_)
        row = pd.DataFrame([[revenue, traffic, staff, sqm, basket if basket > 0 else NAN]], columns=cols)
        row = as_float_frame(row)
        scaled = store_scaler.transform(pd.DataFrame(store_imputer.transform(row), columns=cols))
        cluster = int(store_kmeans.predict(scaled)[0])
        st.success(f"Cluster {cluster}: {SEGMENTS[cluster]}")

# ----------------------------------------------------------------------------- 4. Association rules
with tab_music:
    names = {t: f"{r.item_name} ({t})" for t, r in track_lookup.sort_index().iterrows()}
    with st.form("music_form"):
        played = st.multiselect("Tracks already played", list(names), default=["M08", "M10"], format_func=names.get)
        top_n = st.number_input("How many suggestions", 1, 10, 3)
        go = st.form_submit_button("Recommend")
    if go:
        session = set(played)
        candidates = {}
        for antecedent, consequents in recommendation_lookup.items():
            if session and antecedent.issubset(session):
                for track, conf, lift in consequents:
                    if track not in session and (track not in candidates or conf > candidates[track][0]):
                        candidates[track] = (conf, lift)
        ranked = sorted(candidates.items(), key=lambda kv: kv[1][0], reverse=True)[: int(top_n)]
        if not session:
            st.warning("Pick at least one track.")
        elif not ranked:
            st.info("No mined rule matches this session.")
        for track, (conf, lift) in ranked:
            st.write(f"**{track_lookup.loc[track, 'item_name']}** ({track}): confidence {conf:.1%}, lift {lift:.2f}")
