from __future__ import annotations

from pydantic import BaseModel, Field


class ApplicationRequest(BaseModel):
    age: int = Field(..., ge=18, le=100)
    region: str
    employment_type: str
    monthly_income: float = Field(..., ge=0)
    previous_applications_count: int = Field(..., ge=0)
    previous_rejections_count: int = Field(..., ge=0)
    email_domain: str
    phone_age_days: int = Field(..., ge=0)
    phone_type: str
    requested_amount: float = Field(..., gt=0)
    loan_term_days: int = Field(..., gt=0)
    loan_type: str
    application_hour: int = Field(..., ge=0, le=23)
    application_weekday: int = Field(..., ge=0, le=6)
    device_reuse_count: int = Field(..., ge=0)
    ip_reuse_count: int = Field(..., ge=0)
    proxy_flag: int = Field(..., ge=0, le=1)
    vpn_flag: int = Field(..., ge=0, le=1)
    tor_flag: int = Field(..., ge=0, le=1)
    browser_type: str
    os_type: str
    time_to_fill_application_sec: int = Field(..., ge=0)
    number_of_corrections: int = Field(..., ge=0)
    copy_paste_ratio: float = Field(..., ge=0.0, le=1.0)
    typing_speed: float = Field(..., ge=0.0)
    night_application_flag: int = Field(..., ge=0, le=1)
    weekend_flag: int = Field(..., ge=0, le=1)
    multiple_applications_last_1h: int = Field(..., ge=0)
    multiple_applications_last_24h: int = Field(..., ge=0)
    shared_device_with_other_iin_count: int = Field(..., ge=0)
    shared_ip_with_other_iin_count: int = Field(..., ge=0)
    cluster_id: int = Field(default=-1)
    cluster_size: int = Field(default=1, ge=0)
    cluster_fraud_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    age_income_zscore: float = Field(default=0.0)
    region_income_zscore: float = Field(default=0.0)
    application_velocity_score: float = Field(default=1.0, ge=0.0, le=10.0)
    anomaly_score: float = Field(default=0.0, ge=0.0, le=1.0)
