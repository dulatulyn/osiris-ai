import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = Path(os.environ.get("MODEL_PATH", str(ARTIFACTS_DIR / "v1")))

PIPELINE_PATH = MODEL_PATH / "pipeline.joblib"
FEATURE_META_PATH = MODEL_PATH / "feature_meta.joblib"
METRICS_PATH = MODEL_PATH / "metrics.json"
REGISTRY_PATH = ARTIFACTS_DIR / "registry.json"

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

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

DEFAULT_N_ESTIMATORS = 500
DEFAULT_MAX_DEPTH = 7
DEFAULT_LEARNING_RATE = 0.05
EARLY_STOPPING_ROUNDS = 30
