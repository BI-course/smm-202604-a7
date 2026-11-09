"""
api_consumer_from_dev_flask.py

A frontend-style client that calls each of the four prediction endpoints
exposed by api.py, running on the local Flask development server. Start
api.py first (python api.py), then run this script separately.

This mirrors api.py's four artifacts one-to-one:

    1. Regression        -- /api/v1/models/rent-price-regressor/predictions
    2. Classification     -- /api/v1/models/employee-attrition-classifier/predictions
    3. Clustering         -- /api/v1/models/retail-store-segmenter/predictions
    4. Association Rule   -- /api/v1/models/music-session-recommender/predictions
       Mining (ARM)

Each call uses the same requests.post(..., json=payload, headers=headers)
pattern, the same header set, and the same try/except structure as the
single-endpoint version this file replaces -- only the URL, payload, and the
key read back out of the JSON response differ per endpoint, because each
endpoint returns a different shape (a regression figure, a class + class
probabilities, a cluster id, or a list of recommendations).
"""

import requests

BASE_URL = "http://127.0.0.1:5000"

HEADERS = {
    'Content-Type': 'application/json',  # This header tells the server that the payload is in JSON format
    'Accept': 'application/json',        # This header tells the server that the client expects a JSON response
    'User-Agent': 'ServingMLModels-Client/1.0',  # Identifies the client application (useful for backend logging/debugging)
    # 'Authorization': 'Bearer <token>',  # This can be used for token-based authentication if the API requires it
}


def call_endpoint(path, payload):
    """Send one POST request to the given API path and return the parsed
    JSON response, or None if the request failed. Centralizing the
    try/except here means each of the four calls below gets identical
    error handling without repeating it four times."""
    url = f"{BASE_URL}{path}"
    try:
        response = requests.post(url, json=payload, headers=HEADERS, timeout=10)
        response.raise_for_status()  # Raises an HTTPError if the status code is 4xx or 5xx
        return response.json()

    except requests.exceptions.HTTPError as e:
        print("HTTP error:", e)
        if e.response is not None:
            print("Response body:", e.response.text)

    except requests.exceptions.RequestException as e:
        print("Request failed:", e)

    return None


# 1. REGRESSION -- predict the monthly rent (KES) of a property

regression_payload = {
    "property_type": "Apartment", "neighborhood": "Kilimani",
    "furnishing_status": "Semi-Furnished", "size_sqm": 95,
    "num_bedrooms": 3, "num_bathrooms": 2, "distance_to_cbd_km": 5.5,
    "service_charge_kes": 15000, "garage_size_sqm": 0,
    "security_rating": 4, "amenities_score": 60,
    "nearby_schools_count": 3, "has_garage": 0,
}

result = call_endpoint("/api/v1/models/rent-price-regressor/predictions", regression_payload)
if result is not None:
    print("Predicted monthly rent (KES) =", result["predicted_monthly_rent_kes"])

print("-" * 60)



# 2. CLASSIFICATION -- predict whether an employee is likely to leave (Yes/No)

classification_payload = {
    "department": "Sales", "job_role_level": "Junior",
    "education_level": "Bachelor's", "performance_rating": "Average",
    "years_since_last_promotion": 4, "monthly_salary_kes": 65000,
    "overtime_hours_per_month": 30, "job_satisfaction_score": 2,
    "work_life_balance_score": 2, "distance_from_home_km": 18,
    "num_companies_worked_previously": 4, "remote_work_days_per_week": 0,
    "bonus_amount_kes": None,  # No bonus received: structurally missing
}

result = call_endpoint("/api/v1/models/employee-attrition-classifier/predictions", classification_payload)
if result is not None:
    print("Predicted attrition (Yes = will leave) =", result["predicted_attrition"])
    print("Class probabilities =", result["class_probabilities"])

print("-" * 60)



# 3. CLUSTERING -- assign a retail store to an existing segment

clustering_payload = {
    "monthly_revenue_kes": 1200000,
    "daily_foot_traffic": 400,
    "staff_count": 9,
    "store_size_sqm": 300,
    "avg_basket_size_kes": 120,
}

result = call_endpoint("/api/v1/models/retail-store-segmenter/predictions", clustering_payload)
if result is not None:
    print("Assigned cluster =", result["assigned_cluster"])

print("-" * 60)



# 4. ASSOCIATION RULE MINING -- recommend tracks for a listening session

recommendation_payload = {
    "session": ["M01", "M03"],
    "top_n": 3,
}

result = call_endpoint("/api/v1/models/music-session-recommender/predictions", recommendation_payload)
if result is not None:
    print("Session =", result["session"])
    print("Recommendations =", result["recommendations"])
