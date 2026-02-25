import torch
import joblib
from src.training.model import FraudDetector
from src.training.preprocessing import preprocess_single
from src.config import MODEL_WEIGHTS_PATH, FEATURE_META_PATH, SCALER_PATH, ENCODERS_PATH

meta = joblib.load(FEATURE_META_PATH)
scaler = joblib.load(SCALER_PATH)
encoders = joblib.load(ENCODERS_PATH)

model = FraudDetector(input_dim=meta["input_dim"], feature_cols=meta["feature_cols"])
model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu", weights_only=True))
model.eval()

legit_data = {
    "age": 42, "region": "Almaty", "employment_type": "full_time",
    "monthly_income": 380000, "previous_applications_count": 1,
    "previous_rejections_count": 0, "email_domain": "gmail.com",
    "phone_age_days": 1200, "phone_type": "mobile",
    "requested_amount": 150000, "loan_term_days": 30,
    "loan_type": "microloan", "application_hour": 14,
    "application_weekday": 2, "device_reuse_count": 1,
    "ip_reuse_count": 1, "proxy_flag": 0, "vpn_flag": 0, "tor_flag": 0,
    "browser_type": "Chrome", "os_type": "Android",
    "time_to_fill_application_sec": 450, "number_of_corrections": 2,
    "copy_paste_ratio": 0.05, "typing_speed": 4.5,
    "night_application_flag": 0, "weekend_flag": 0,
    "multiple_applications_last_1h": 0, "multiple_applications_last_24h": 0,
    "shared_device_with_other_iin_count": 0,
    "shared_ip_with_other_iin_count": 0,
    "cluster_id": -1, "cluster_size": 1,
    "cluster_fraud_ratio": 0.0, "age_income_zscore": 0.5,
    "region_income_zscore": 0.3, "application_velocity_score": 1.5,
    "anomaly_score": 0.01,
}

X = preprocess_single(legit_data, scaler=scaler, encoders=encoders, feature_cols=meta["feature_cols"])
tensor = torch.tensor(X, dtype=torch.float32)
with torch.no_grad():
    logit = model(tensor)
    prob = torch.sigmoid(logit).item()
print(f"Legit mock -> probability: {prob:.6f}, score: {int(round(prob * 100))}")

fraud_data = dict(legit_data)
fraud_data.update({
    "age": 19, "phone_age_days": 15, "email_domain": "tempmail.com",
    "device_reuse_count": 8, "ip_reuse_count": 12,
    "proxy_flag": 1, "tor_flag": 1, "vpn_flag": 1,
    "time_to_fill_application_sec": 30, "copy_paste_ratio": 0.85,
    "typing_speed": 10.5, "night_application_flag": 1,
    "multiple_applications_last_1h": 4, "multiple_applications_last_24h": 10,
    "shared_device_with_other_iin_count": 8,
    "shared_ip_with_other_iin_count": 6,
    "application_hour": 2, "application_weekday": 6,
})

X2 = preprocess_single(fraud_data, scaler=scaler, encoders=encoders, feature_cols=meta["feature_cols"])
tensor2 = torch.tensor(X2, dtype=torch.float32)
with torch.no_grad():
    logit2 = model(tensor2)
    prob2 = torch.sigmoid(logit2).item()
print(f"Fraud mock -> probability: {prob2:.6f}, score: {int(round(prob2 * 100))}")

print(f"\nThreshold from meta: {meta.get('threshold', 0.5):.3f}")
