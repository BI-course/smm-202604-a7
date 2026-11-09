import gradio as gr
import joblib
import numpy as np
import pandas as pd
import spaces


@spaces.GPU
def _warm_zerogpu():
    return None


_warm_zerogpu()

regression_model = joblib.load("rent_price_model.joblib")
classification_pipeline = joblib.load("logisticregression_classifier_for_employee_attrition.joblib")

clustering_artifact = joblib.load("kmeans_retail_store_segmentation.joblib")
clustering_imputer = clustering_artifact["imputer"]
clustering_scaler = clustering_artifact["scaler"]
clustering_kmeans = clustering_artifact["kmeans"]

arm_artifact = joblib.load("apriori_music_session_rules.joblib")
recommendation_lookup = arm_artifact["recommendation_lookup"]
track_lookup = arm_artifact["track_lookup"]


def _num(value):
    """Gradio Number fields give None when left blank; the pipelines' imputers
    were fitted on float columns with NaN as the missing marker."""
    return np.nan if value is None else float(value)


# -----------------------------------------------------------------------
# 1. Regression -- predict the monthly rent (KES) of a property
# -----------------------------------------------------------------------
def predict_rent(property_type, neighborhood, furnishing_status, size_sqm,
                 num_bedrooms, num_bathrooms, distance_to_cbd_km,
                 service_charge_kes, garage_size_sqm, security_rating,
                 amenities_score, nearby_schools_count, has_garage):
    input_row = pd.DataFrame([{
        "property_type": property_type, "neighborhood": neighborhood,
        "furnishing_status": furnishing_status, "size_sqm": _num(size_sqm),
        "num_bedrooms": _num(num_bedrooms), "num_bathrooms": _num(num_bathrooms),
        "distance_to_cbd_km": _num(distance_to_cbd_km),
        "service_charge_kes": _num(service_charge_kes),
        "garage_size_sqm": _num(garage_size_sqm),
        "security_rating": _num(security_rating),
        "amenities_score": _num(amenities_score),
        "nearby_schools_count": _num(nearby_schools_count),
        "has_garage": 1.0 if has_garage == "Yes" else 0.0,
    }])

    predicted_rent_kes = float(regression_model.predict(input_row)[0])
    return f"KES {predicted_rent_kes:,.2f} / month"


# -----------------------------------------------------------------------
# 2. Classification -- predict employee attrition (Yes = likely to leave)
# -----------------------------------------------------------------------
def predict_attrition(department, job_role_level, education_level, performance_rating,
                      years_since_last_promotion, monthly_salary_kes,
                      overtime_hours_per_month, job_satisfaction_score,
                      work_life_balance_score, distance_from_home_km,
                      num_companies_worked_previously, remote_work_days_per_week,
                      bonus_amount_kes):
    input_row = pd.DataFrame([{
        "department": department, "job_role_level": job_role_level,
        "education_level": education_level, "performance_rating": performance_rating,
        "years_since_last_promotion": _num(years_since_last_promotion),
        "monthly_salary_kes": _num(monthly_salary_kes),
        "overtime_hours_per_month": _num(overtime_hours_per_month),
        "job_satisfaction_score": _num(job_satisfaction_score),
        "work_life_balance_score": _num(work_life_balance_score),
        "distance_from_home_km": _num(distance_from_home_km),
        "num_companies_worked_previously": _num(num_companies_worked_previously),
        "remote_work_days_per_week": _num(remote_work_days_per_week),
        "bonus_amount_kes": _num(bonus_amount_kes or None),
    }])
    predicted_class = classification_pipeline.predict(input_row)[0]
    predicted_probabilities = classification_pipeline.predict_proba(input_row)[0]
    class_labels = classification_pipeline.named_steps["model"].classes_
    probability_lines = "\n".join(
        f"  {label}: {prob:.1%}" for label, prob in zip(class_labels, predicted_probabilities)
    )
    verdict = "likely to LEAVE" if predicted_class == "Yes" else "likely to STAY"
    return f"Predicted attrition: {predicted_class} ({verdict})\n\n{probability_lines}"


# -----------------------------------------------------------------------
# 3. Clustering -- assign a retail store to an existing segment
# -----------------------------------------------------------------------
def predict_segment(monthly_revenue_kes, daily_foot_traffic, staff_count,
                    store_size_sqm, avg_basket_size_kes):
    columns = list(clustering_scaler.feature_names_in_)
    input_row = pd.DataFrame([{
        "monthly_revenue_kes": _num(monthly_revenue_kes),
        "daily_foot_traffic": _num(daily_foot_traffic),
        "staff_count": _num(staff_count),
        "store_size_sqm": _num(store_size_sqm),
        "avg_basket_size_kes": _num(avg_basket_size_kes),
    }])[columns]
    imputed = pd.DataFrame(clustering_imputer.transform(input_row), columns=columns)
    scaled_input = clustering_scaler.transform(imputed)
    cluster = int(clustering_kmeans.predict(scaled_input)[0])
    return f"Assigned segment: Cluster {cluster}"


# -----------------------------------------------------------------------
# 4. Association Rule Mining -- recommend tracks for a listening session
# -----------------------------------------------------------------------
def recommend_tracks(session_text, top_n):
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
        return "No recommendations for this session -- try a track combination seen in the training sessions (e.g. M01, M03)."

    lines = [
        f"{track_lookup.loc[track_id, 'item_name']} ({track_id}, {track_lookup.loc[track_id, 'category']})  --  "
        f"confidence {confidence:.1%}, lift {lift:.2f}"
        for track_id, (confidence, lift) in ranked
    ]
    return "\n".join(lines)


with gr.Blocks(title="BBT 4206 Model Validation Demos") as demo:
    gr.Markdown("# BBT 4206 -- Model Validation Demos\nSwitch tabs to try a different model.")

    with gr.Tab("Rent Price Regressor"):
        gr.Interface(
            fn=predict_rent,
            inputs=[
                gr.Dropdown(label="Property Type",
                            choices=["Apartment", "Bungalow", "Maisonette", "Studio", "Townhouse"],
                            value="Apartment"),
                gr.Dropdown(label="Neighborhood",
                            choices=["Karen", "Kilimani", "Ngong Road", "Ruaka", "South B", "Westlands"],
                            value="Kilimani"),
                gr.Dropdown(label="Furnishing Status",
                            choices=["Unfurnished", "Semi-Furnished", "Fully Furnished"],
                            value="Semi-Furnished"),
                gr.Number(label="Size (sqm)", value=95),
                gr.Number(label="Bedrooms", value=3),
                gr.Number(label="Bathrooms", value=2),
                gr.Number(label="Distance to CBD (km)", value=5.5),
                gr.Number(label="Service Charge (KES)", value=15000),
                gr.Number(label="Garage Size (sqm)", value=0),
                gr.Number(label="Security Rating (1-5)", value=4),
                gr.Number(label="Amenities Score (0-100)", value=60),
                gr.Number(label="Nearby Schools", value=3),
                gr.Radio(label="Has Garage", choices=["No", "Yes"], value="No"),
            ],
            outputs=gr.Textbox(label="Predicted Monthly Rent"),
        )

    with gr.Tab("Employee Attrition Classifier"):
        gr.Interface(
            fn=predict_attrition,
            inputs=[
                gr.Dropdown(label="Department",
                            choices=["Engineering", "Finance", "HR", "Operations", "Sales"],
                            value="Sales"),
                gr.Dropdown(label="Job Role Level",
                            choices=["Junior", "Mid", "Senior", "Lead"], value="Junior"),
                gr.Dropdown(label="Education Level",
                            choices=["Diploma", "Bachelor's", "Master's", "PhD"],
                            value="Bachelor's"),
                gr.Dropdown(label="Performance Rating",
                            choices=["Low", "Average", "High", "Outstanding"],
                            value="Average"),
                gr.Number(label="Years Since Last Promotion", value=4),
                gr.Number(label="Monthly Salary (KES)", value=65000),
                gr.Number(label="Overtime Hours per Month", value=30),
                gr.Number(label="Job Satisfaction Score (1-5)", value=2),
                gr.Number(label="Work-Life Balance Score (1-5)", value=2),
                gr.Number(label="Distance From Home (km)", value=18),
                gr.Number(label="Previous Companies Worked", value=4),
                gr.Number(label="Remote Work Days per Week", value=0),
                gr.Number(label="Bonus Amount (KES) (leave 0 if none)", value=0),
            ],
            outputs=gr.Textbox(label="Predicted Attrition"),
        )

    with gr.Tab("Retail Store Segmenter"):
        gr.Interface(
            fn=predict_segment,
            inputs=[
                gr.Number(label="Monthly Revenue (KES)", value=1200000),
                gr.Number(label="Daily Foot Traffic", value=400),
                gr.Number(label="Staff Count", value=9),
                gr.Number(label="Store Size (sqm)", value=300),
                gr.Number(label="Average Basket Size (KES)", value=120),
            ],
            outputs=gr.Textbox(label="Prediction"),
        )

    with gr.Tab("Music Session Recommender"):
        gr.Interface(
            fn=recommend_tracks,
            inputs=[
                gr.Textbox(label="Session (comma-separated track ids)", value="M01, M03"),
                gr.Number(label="Top N Recommendations", value=3),
            ],
            outputs=gr.Textbox(label="Recommendations"),
        )

if __name__ == "__main__":
    demo.launch()