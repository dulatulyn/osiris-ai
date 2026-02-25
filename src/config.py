import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ARTIFACTS_DIR / "v1")))

MODEL_WEIGHTS_PATH = MODEL_PATH / "fraud_model.pt"
SCALER_PATH = MODEL_PATH / "scaler.joblib"
ENCODERS_PATH = MODEL_PATH / "encoders.joblib"
FEATURE_META_PATH = MODEL_PATH / "feature_meta.joblib"
METRICS_PATH = MODEL_PATH / "metrics.json"
DEFAULT_DATASET_PATH = DATA_DIR / "dataset.csv"

TARGET_COLUMN = "is_fraud"

DROP_COLUMNS = [
    "iin_hash",
    "device_id_hash",
    "ip_address_hash",
    "gender",
    "cluster_fraud_ratio",
    "anomaly_score",
]

CATEGORICAL_COLUMNS = [
    "region",
    "employment_type",
    "email_domain",
    "phone_type",
    "loan_type",
    "browser_type",
    "os_type",
]

NUMERIC_COLUMNS = [
    "age",
    "monthly_income",
    "previous_applications_count",
    "previous_rejections_count",
    "phone_age_days",
    "requested_amount",
    "loan_term_days",
    "application_hour",
    "application_weekday",
    "device_reuse_count",
    "ip_reuse_count",
    "proxy_flag",
    "vpn_flag",
    "tor_flag",
    "time_to_fill_application_sec",
    "number_of_corrections",
    "copy_paste_ratio",
    "typing_speed",
    "night_application_flag",
    "weekend_flag",
    "multiple_applications_last_1h",
    "multiple_applications_last_24h",
    "shared_device_with_other_iin_count",
    "shared_ip_with_other_iin_count",
    "cluster_id",
    "cluster_size",
    "age_income_zscore",
    "region_income_zscore",
    "application_velocity_score",
]

HIDDEN_DIMS = [128, 64, 32]
DROPOUT_RATES = [0.3, 0.2, 0.1]

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 512
DEFAULT_LR = 1e-3
DEFAULT_WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 10
