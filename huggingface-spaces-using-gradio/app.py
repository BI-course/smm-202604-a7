import streamlit as st
import joblib
import numpy as np
import pandas as pd
import os


def load_artifacts():
    MODEL_DIR = "./model"

    regression_model = joblib.load(
        os.path.join(MODEL_DIR, "rent_price_model.joblib")
    )

    classification_pipeline = joblib.load(
        os.path.join(MODEL_DIR, "logisticregression_classifier_for_employee_attrition.joblib")
    )

    clustering_artifact = joblib.load(
        os.path.join(MODEL_DIR, "kmeans_retail_store_segmentation.joblib")
    )

    arm_artifact = joblib.load(
        os.path.join(MODEL_DIR, "apriori_music_session_rules.joblib")
    )
    return regression_model, classification_pipeline, clustering_artifact, arm_artifact


regression_model, classification_pipeline, clustering_artifact, arm_artifact = load_artifacts()
clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]

st.set_page_config(page_title="BBT 4206 Model Validation Demos", page_icon="\U0001F4CA", layout="centered")
st.title("BBT 4206 -- Model Validation Demos")

tab_regression, tab_classification, tab_clustering, tab_arm = st.tabs([
    "Rent Price Regressor", "Employee Attrition Classifier",
    "Retail Store Segmenter", "Music Session Recommender",
])

# -----------------------------------------------------------------------
# 1. Regression -- predict the monthly rent (KES) of a property
# -----------------------------------------------------------------------
with tab_regression:
    with st.form("regression_form"):
        property_type = st.selectbox(
            "Property Type", ["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"])
        neighborhood = st.selectbox(
            "Neighborhood", ["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"],
            index=1)

        furnishing_status = st.selectbox(
            "Furnishing Status", ["Unfurnished", "Semi-Furnished", "Fully Furnished"], index=1)
        size_sqm = st.number_input("Size (sqm)", value=95.0)
        num_bedrooms = st.number_input("Bedrooms", value=3)
        num_bathrooms = st.number_input("Bathrooms", value=2)
        distance_to_cbd_km = st.number_input("Distance to CBD (km)", value=5.5)
        service_charge_kes = st.number_input("Service Charge (KES)", value=15000)
        garage_size_sqm = st.number_input("Garage Size (sqm)", value=0.0)
        security_rating = st.number_input("Security Rating (1-5)", value=4.0)
        amenities_score = st.number_input("Amenities Score (0-100)", value=60.0)
        nearby_schools_count = st.number_input("Nearby Schools", value=3)
        has_garage = st.selectbox("Has Garage", ["No", "Yes"])
        submitted = st.form_submit_button("Predict Rent")

    if submitted:
        input_row = pd.DataFrame([{
            "property_type": property_type, "neighborhood": neighborhood,
            "furnishing_status": furnishing_status, "size_sqm": float(size_sqm),
            "num_bedrooms": float(num_bedrooms), "num_bathrooms": float(num_bathrooms),
            "distance_to_cbd_km": float(distance_to_cbd_km),
            "service_charge_kes": float(service_charge_kes),
            "garage_size_sqm": float(garage_size_sqm),
            "security_rating": float(security_rating),
            "amenities_score": float(amenities_score),
            "nearby_schools_count": float(nearby_schools_count),
            "has_garage": 1.0 if has_garage == "Yes" else 0.0,
        }])

        predicted_rent_kes = float(regression_model.predict(input_row)[0])
        st.success(f"Predicted monthly rent: KES {predicted_rent_kes:,.2f}")

# -----------------------------------------------------------------------
# 2. Classification -- predict employee attrition (Yes = likely to leave)
# -----------------------------------------------------------------------
with tab_classification:
    with st.form("classification_form"):
        department = st.selectbox(
            "Department", ["Engineering", "Finance", "HR", "Operations", "Sales"], index=4)
        # The three ordinal columns below use the exact category sets the
        # pipeline's OrdinalEncoder was fitted on.
        job_role_level = st.selectbox("Job Role Level", ["Junior", "Mid", "Senior", "Lead"])
        education_level = st.selectbox(
            "Education Level", ["Diploma", "Bachelor's", "Master's", "PhD"], index=1)
        performance_rating = st.selectbox(
            "Performance Rating", ["Low", "Average", "High", "Outstanding"], index=1)
        years_since_last_promotion = st.number_input("Years Since Last Promotion", value=4.0)
        monthly_salary_kes = st.number_input("Monthly Salary (KES)", value=65000)
        overtime_hours_per_month = st.number_input("Overtime Hours per Month", value=30)
        job_satisfaction_score = st.number_input("Job Satisfaction Score (1-5)", value=2.0)
        work_life_balance_score = st.number_input("Work-Life Balance Score (1-5)", value=2.0)
        distance_from_home_km = st.number_input("Distance From Home (km)", value=18.0)
        num_companies_worked_previously = st.number_input("Previous Companies Worked", value=4)
        remote_work_days_per_week = st.number_input("Remote Work Days per Week", value=0)
        bonus_amount_kes = st.number_input("Bonus Amount (KES) (leave 0 if none)", value=0)
        submitted2 = st.form_submit_button("Predict Attrition")

    if submitted2:
        input_row = pd.DataFrame([{
            "department": department, "job_role_level": job_role_level,
            "education_level": education_level, "performance_rating": performance_rating,
            "years_since_last_promotion": float(years_since_last_promotion),
            "monthly_salary_kes": float(monthly_salary_kes),
            "overtime_hours_per_month": float(overtime_hours_per_month),
            "job_satisfaction_score": float(job_satisfaction_score),
            "work_life_balance_score": float(work_life_balance_score),
            "distance_from_home_km": float(distance_from_home_km),
            "num_companies_worked_previously": float(num_companies_worked_previously),
            "remote_work_days_per_week": float(remote_work_days_per_week),
            # 0 / blank means "no bonus received" -> missing, as in training.
            "bonus_amount_kes": float(bonus_amount_kes) if bonus_amount_kes else np.nan,
        }])
        predicted_class = classification_pipeline.predict(input_row)[0]
        predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
        class_labels = classification_pipeline.named_steps["model"].classes_
        verdict = "likely to LEAVE" if predicted_class == "Yes" else "likely to STAY"
        st.success(f"Predicted attrition: {predicted_class} ({verdict})")
        st.write({str(label): round(float(prob), 4)
                  for label, prob in zip(class_labels, predicted_probabilities)})

# -----------------------------------------------------------------------
# 3. Clustering -- assign a retail store to an existing segment
# -----------------------------------------------------------------------
with tab_clustering:
    with st.form("clustering_form"):
        monthly_revenue_kes = st.number_input("Monthly Revenue (KES)", min_value=0.0, value=1200000.0)
        daily_foot_traffic = st.number_input("Daily Foot Traffic", min_value=0.0, value=400.0)
        staff_count = st.number_input("Staff Count", min_value=0.0, value=9.0)
        store_size_sqm = st.number_input("Store Size (sqm)", min_value=0.0, value=300.0)
        avg_basket_size_kes = st.number_input("Average Basket Size (KES)", min_value=0.0, value=120.0)
        submitted3 = st.form_submit_button("Predict Segment")

    if submitted3:
        columns = list(clustering_scaler.feature_names_in_)
        input_row = pd.DataFrame([{
            "monthly_revenue_kes": monthly_revenue_kes,
            "daily_foot_traffic": daily_foot_traffic,
            "staff_count": staff_count,
            "store_size_sqm": store_size_sqm,
            "avg_basket_size_kes": avg_basket_size_kes,
        }])[columns]
        imputed = pd.DataFrame(clustering_imputer.transform(input_row), columns=columns)
        scaled_input = clustering_scaler.transform(imputed)
        cluster = int(clustering_kmeans.predict(scaled_input)[0])
        st.success(f"Assigned segment: Cluster {cluster}")

# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend tracks for a listening session
# -----------------------------------------------------------------------
with tab_arm:
    with st.form("arm_form"):
        session_text = st.text_input("Session (comma-separated track ids)", "M01, M03")
        top_n = st.number_input("Top N Recommendations", min_value=1, value=3)
        submitted4 = st.form_submit_button("Get Recommendations")

    if submitted4:
        session = {t.strip().upper() for t in session_text.split(",") if t.strip()}
        candidates = {}
        for antecedent, consequent_list in recommendation_lookup.items():
            if antecedent.issubset(session):
                for track_id, confidence, lift in consequent_list:
                    if track_id not in session:
                        if track_id not in candidates or confidence > candidates[track_id][0]:
                            candidates[track_id] = (confidence, lift)

        ranked = sorted(candidates.items(), key=lambda item: item[1][0], reverse=True)[:int(top_n)]
        if not ranked:
            st.warning(
                "No recommendations for this session -- try a track combination seen in the training sessions (e.g. M01, M03).")
        else:
            for track_id, (confidence, lift) in ranked:
                st.write(f"**{track_lookup.loc[track_id, 'item_name']}** ({track_id}, "
                         f"{track_lookup.loc[track_id, 'category']}) "
                         f"-- confidence {confidence:.1%}, lift {lift:.2f}")
