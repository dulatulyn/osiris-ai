from __future__ import annotations

import argparse
import calendar
import hashlib
import os
import random
import sys
from datetime import date, datetime, timedelta
from typing import Any

import numpy as np
import pandas as pd

KAZAKHSTAN_REGIONS = [
    "Almaty",
    "Astana",
    "Shymkent",
    "Akmola",
    "Aktobe",
    "Almaty region",
    "Atyrau",
    "East Kazakhstan",
    "Zhambyl",
    "West Kazakhstan",
    "Karaganda",
    "Kostanay",
    "Kyzylorda",
    "Mangystau",
    "North Kazakhstan",
    "Pavlodar",
    "Turkistan",
    "Ulytau",
    "Zhetysu",
    "Abai",
]

REGION_WEIGHTS = [
    0.15,
    0.12,
    0.06,
    0.04,
    0.04,
    0.05,
    0.04,
    0.05,
    0.04,
    0.03,
    0.06,
    0.04,
    0.03,
    0.04,
    0.03,
    0.04,
    0.04,
    0.02,
    0.04,
    0.04,
]

EMPLOYMENT_TYPES = ["full_time", "part_time", "self_employed", "unemployed", "student"]
EMPLOYMENT_WEIGHTS_LEGIT = [0.50, 0.15, 0.20, 0.05, 0.10]
EMPLOYMENT_WEIGHTS_FRAUD = [0.20, 0.15, 0.15, 0.30, 0.20]

EMAIL_DOMAINS_LEGIT = ["gmail.com", "mail.ru", "yandex.ru", "yahoo.com", "outlook.com", "icloud.com"]
EMAIL_WEIGHTS_LEGIT = [0.40, 0.25, 0.15, 0.08, 0.07, 0.05]

EMAIL_DOMAINS_FRAUD = [
    "gmail.com", "mail.ru", "tempmail.com", "guerrillamail.com",
    "throwaway.email", "10minutemail.com", "yandex.ru",
]
EMAIL_WEIGHTS_FRAUD = [0.15, 0.10, 0.25, 0.20, 0.15, 0.10, 0.05]

PHONE_TYPES = ["mobile", "landline"]
LOAN_TYPES = ["microloan", "consumer", "payday"]
BROWSER_TYPES = ["Chrome", "Firefox", "Safari", "Edge", "Opera"]
OS_TYPES = ["Windows", "iOS", "Android", "macOS", "Linux"]

REGION_INCOME_MAP: dict[str, tuple[float, float]] = {
    "Almaty": (350_000, 120_000),
    "Astana": (380_000, 130_000),
    "Shymkent": (220_000, 80_000),
    "Akmola": (230_000, 85_000),
    "Aktobe": (260_000, 90_000),
    "Almaty region": (240_000, 90_000),
    "Atyrau": (400_000, 150_000),
    "East Kazakhstan": (250_000, 90_000),
    "Zhambyl": (200_000, 70_000),
    "West Kazakhstan": (280_000, 100_000),
    "Karaganda": (270_000, 95_000),
    "Kostanay": (230_000, 80_000),
    "Kyzylorda": (210_000, 75_000),
    "Mangystau": (350_000, 130_000),
    "North Kazakhstan": (220_000, 80_000),
    "Pavlodar": (260_000, 90_000),
    "Turkistan": (190_000, 65_000),
    "Ulytau": (240_000, 85_000),
    "Zhetysu": (220_000, 80_000),
    "Abai": (230_000, 85_000),
}

def _iin_check_digit(digits_11: list[int]) -> int:
    weights_1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    s = sum(d * w for d, w in zip(digits_11, weights_1))
    remainder = s % 11
    if remainder < 10:
        return remainder

    weights_2 = [3, 4, 5, 6, 7, 8, 9, 10, 11, 1, 2]
    s2 = sum(d * w for d, w in zip(digits_11, weights_2))
    remainder2 = s2 % 11
    return remainder2 if remainder2 < 10 else 0

def generate_iin(birth_date: date, gender: str) -> str:
    yy = birth_date.year % 100
    mm = birth_date.month
    dd = birth_date.day

    century = birth_date.year // 100
    if century == 19:
        g = 1 if gender == "male" else 2
    elif century == 20:
        g = 3 if gender == "male" else 4
    else:
        g = 5 if gender == "male" else 6

    serial = random.randint(0, 9999)

    prefix = f"{yy:02d}{mm:02d}{dd:02d}{g}{serial:04d}"
    digits_11 = [int(c) for c in prefix]
    check = _iin_check_digit(digits_11)
    return prefix + str(check)

def _sha256_short(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:16]

def _generate_income(region: str, employment: str, age: int, is_fraud: bool) -> float:
    mean, std = REGION_INCOME_MAP.get(region, (250_000, 90_000))

    emp_multiplier = {
        "full_time": 1.0,
        "part_time": 0.55,
        "self_employed": 1.1,
        "unemployed": 0.15,
        "student": 0.20,
    }[employment]

    age_multiplier = 1.0
    if age < 22:
        age_multiplier = 0.5
    elif age < 30:
        age_multiplier = 0.85
    elif age < 45:
        age_multiplier = 1.15
    elif age < 55:
        age_multiplier = 1.05
    else:
        age_multiplier = 0.80

    income = max(30_000, np.random.normal(mean * emp_multiplier * age_multiplier, std * 0.5))

    if is_fraud:
        if random.random() < 0.4:
            income *= random.uniform(1.8, 4.0)

    return round(income, -3)

def _generate_legit_row(
    idx: int,
    rng: np.random.Generator,
    device_pool: list[str],
    ip_pool: list[str],
) -> dict[str, Any]:
    age = int(np.clip(rng.normal(35, 10), 18, 70))
    gender = rng.choice(["male", "female"])
    birth_year = 2026 - age
    birth_month = rng.integers(1, 13)
    max_day = calendar.monthrange(birth_year, birth_month)[1]
    birth_day = rng.integers(1, max_day + 1)
    birth_date = date(birth_year, birth_month, birth_day)

    iin = generate_iin(birth_date, gender)
    region = rng.choice(KAZAKHSTAN_REGIONS, p=REGION_WEIGHTS)
    employment = rng.choice(EMPLOYMENT_TYPES, p=EMPLOYMENT_WEIGHTS_LEGIT)
    income = _generate_income(region, employment, age, is_fraud=False)

    prev_apps = int(np.clip(rng.poisson(2), 0, 20))
    prev_rejections = int(np.clip(rng.binomial(prev_apps, 0.15), 0, prev_apps))

    email_domain = rng.choice(EMAIL_DOMAINS_LEGIT, p=EMAIL_WEIGHTS_LEGIT)
    phone_age = int(np.clip(rng.normal(700, 500), 1, 3650))
    phone_type = "mobile" if rng.random() < 0.92 else "landline"

    if rng.random() < 0.4:
        loan_type = "microloan"
        requested_amount = float(rng.integers(20_000, 300_001))
        loan_term = int(rng.choice([7, 14, 21, 30]))
    elif rng.random() < 0.6:
        loan_type = "consumer"
        requested_amount = float(rng.integers(200_000, 5_000_001))
        loan_term = int(rng.choice([90, 180, 365, 730, 1095]))
    else:
        loan_type = "payday"
        requested_amount = float(rng.integers(10_000, 150_001))
        loan_term = int(rng.choice([7, 14, 30]))

    if rng.random() < 0.70:
        app_hour = int(rng.integers(8, 21))
    else:
        app_hour = int(rng.integers(0, 24))
    app_weekday = int(rng.integers(0, 7))

    device_id = rng.choice(device_pool)
    device_reuse = int(np.clip(rng.poisson(1.5), 1, 15))
    ip = rng.choice(ip_pool)
    ip_reuse = int(np.clip(rng.poisson(2.0), 1, 15))

    proxy = 1 if rng.random() < 0.06 else 0
    vpn = 1 if rng.random() < 0.15 else 0
    tor = 1 if rng.random() < 0.01 else 0
    browser = str(rng.choice(BROWSER_TYPES, p=[0.55, 0.15, 0.15, 0.10, 0.05]))
    os_type = str(rng.choice(OS_TYPES, p=[0.25, 0.20, 0.35, 0.15, 0.05]))

    fill_time = int(np.clip(rng.normal(400, 180), 30, 1800))
    corrections = int(np.clip(rng.poisson(3), 0, 20))
    copy_paste = round(float(np.clip(rng.beta(2, 5), 0, 0.8)), 3)
    typing_speed = round(float(np.clip(rng.normal(5.0, 1.8), 1, 12)), 2)
    night_flag = 1 if 0 <= app_hour < 6 else 0
    weekend_flag = 1 if app_weekday >= 5 else 0
    multi_1h = int(np.clip(rng.poisson(0.3), 0, 5))
    multi_24h = int(np.clip(rng.poisson(0.8), 0, 8))

    shared_device_iins = int(np.clip(rng.poisson(0.5), 0, 6))
    shared_ip_iins = int(np.clip(rng.poisson(0.8), 0, 8))

    return {
        "is_fraud": 0,
        "iin_hash": _sha256_short(iin),
        "age": age,
        "gender": gender,
        "region": region,
        "employment_type": employment,
        "monthly_income": income,
        "previous_applications_count": prev_apps,
        "previous_rejections_count": prev_rejections,
        "email_domain": email_domain,
        "phone_age_days": phone_age,
        "phone_type": phone_type,
        "requested_amount": requested_amount,
        "loan_term_days": loan_term,
        "loan_type": loan_type,
        "application_hour": app_hour,
        "application_weekday": app_weekday,
        "device_id_hash": _sha256_short(device_id),
        "device_reuse_count": device_reuse,
        "ip_address_hash": _sha256_short(ip),
        "ip_reuse_count": ip_reuse,
        "proxy_flag": proxy,
        "vpn_flag": vpn,
        "tor_flag": tor,
        "browser_type": browser,
        "os_type": os_type,
        "time_to_fill_application_sec": fill_time,
        "number_of_corrections": corrections,
        "copy_paste_ratio": copy_paste,
        "typing_speed": typing_speed,
        "night_application_flag": night_flag,
        "weekend_flag": weekend_flag,
        "multiple_applications_last_1h": multi_1h,
        "multiple_applications_last_24h": multi_24h,
        "shared_device_with_other_iin_count": shared_device_iins,
        "shared_ip_with_other_iin_count": shared_ip_iins,
    }

def _generate_fraud_row(
    idx: int,
    rng: np.random.Generator,
    device_pool: list[str],
    ip_pool: list[str],
    fraud_device_pool: list[str],
    fraud_ip_pool: list[str],
) -> dict[str, Any]:
    is_sophisticated = rng.random() < 0.35

    if rng.random() < 0.3:
        age = int(rng.integers(18, 24))
    elif rng.random() < 0.2:
        age = int(rng.integers(55, 71))
    else:
        age = int(np.clip(rng.normal(30, 8), 18, 70))

    gender = rng.choice(["male", "female"])
    birth_year = 2026 - age
    birth_month = rng.integers(1, 13)
    max_day = calendar.monthrange(birth_year, birth_month)[1]
    birth_day = rng.integers(1, max_day + 1)
    birth_date = date(birth_year, birth_month, birth_day)

    iin = generate_iin(birth_date, gender)
    region = rng.choice(KAZAKHSTAN_REGIONS, p=REGION_WEIGHTS)
    employment = rng.choice(EMPLOYMENT_TYPES, p=EMPLOYMENT_WEIGHTS_FRAUD)
    income = _generate_income(region, employment, age, is_fraud=True)

    prev_apps = int(np.clip(rng.poisson(4), 0, 25))
    prev_rejections = int(np.clip(rng.binomial(prev_apps, 0.45), 0, prev_apps))

    if is_sophisticated:
        email_domain = rng.choice(EMAIL_DOMAINS_LEGIT, p=EMAIL_WEIGHTS_LEGIT)
    else:
        email_domain = rng.choice(EMAIL_DOMAINS_FRAUD, p=EMAIL_WEIGHTS_FRAUD)
    phone_age = int(np.clip(rng.exponential(150), 1, 1200))
    phone_type = "mobile" if rng.random() < 0.97 else "landline"

    if rng.random() < 0.5:
        loan_type = "payday"
        requested_amount = float(rng.integers(80_000, 150_001))
        loan_term = int(rng.choice([7, 14, 30]))
    elif rng.random() < 0.5:
        loan_type = "microloan"
        requested_amount = float(rng.integers(150_000, 500_001))
        loan_term = int(rng.choice([14, 30]))
    else:
        loan_type = "consumer"
        requested_amount = float(rng.integers(1_000_000, 10_000_001))
        loan_term = int(rng.choice([180, 365, 730]))

    if is_sophisticated:
        if rng.random() < 0.65:
            app_hour = int(rng.integers(8, 21))
        else:
            app_hour = int(rng.integers(0, 24))
        app_weekday = int(rng.integers(0, 7))
    else:
        if rng.random() < 0.40:
            app_hour = int(rng.integers(0, 6))
        elif rng.random() < 0.3:
            app_hour = int(rng.integers(22, 24))
        else:
            app_hour = int(rng.integers(0, 24))
        app_weekday = int(rng.choice([0, 1, 2, 3, 4, 5, 6], p=[0.10, 0.10, 0.10, 0.10, 0.10, 0.25, 0.25]))

    if is_sophisticated:
        device_id = rng.choice(device_pool)
        device_reuse = int(np.clip(rng.poisson(2.5), 1, 12))
        ip = rng.choice(ip_pool)
        ip_reuse = int(np.clip(rng.poisson(3.0), 1, 15))
    else:
        use_fraud_device = rng.random() < 0.65
        device_id = rng.choice(fraud_device_pool) if use_fraud_device else rng.choice(device_pool)
        device_reuse = int(np.clip(rng.poisson(5), 1, 30))
        use_fraud_ip = rng.random() < 0.60
        ip = rng.choice(fraud_ip_pool) if use_fraud_ip else rng.choice(ip_pool)
        ip_reuse = int(np.clip(rng.poisson(6), 1, 35))

    if is_sophisticated:
        proxy = 1 if rng.random() < 0.10 else 0
        vpn = 1 if rng.random() < 0.20 else 0
        tor = 0
    else:
        proxy = 1 if rng.random() < 0.30 else 0
        vpn = 1 if rng.random() < 0.45 else 0
        tor = 1 if rng.random() < 0.12 else 0

    browser = str(rng.choice(BROWSER_TYPES, p=[0.45, 0.20, 0.05, 0.10, 0.20]))
    os_type = str(rng.choice(OS_TYPES, p=[0.35, 0.05, 0.30, 0.05, 0.25]))

    if is_sophisticated:
        fill_time = int(np.clip(rng.normal(320, 150), 60, 1200))
        copy_paste = round(float(np.clip(rng.beta(2.5, 4), 0, 0.7)), 3)
        typing_speed = round(float(np.clip(rng.normal(5.5, 1.8), 1.5, 11)), 2)
    else:
        if rng.random() < 0.5:
            fill_time = int(np.clip(rng.exponential(80), 15, 300))
        else:
            fill_time = int(np.clip(rng.normal(220, 100), 30, 600))
        copy_paste = round(float(np.clip(rng.beta(4, 3), 0.05, 0.95)), 3)
        typing_speed = round(float(np.clip(rng.normal(7.0, 2.0), 2, 14)), 2)

    corrections = int(np.clip(rng.poisson(5), 0, 25))
    night_flag = 1 if 0 <= app_hour < 6 else 0
    weekend_flag = 1 if app_weekday >= 5 else 0

    if is_sophisticated:
        multi_1h = int(np.clip(rng.poisson(1.0), 0, 6))
        multi_24h = int(np.clip(rng.poisson(2.0), 0, 10))
        shared_device_iins = int(np.clip(rng.poisson(1.5), 0, 8))
        shared_ip_iins = int(np.clip(rng.poisson(2.0), 0, 10))
    else:
        multi_1h = int(np.clip(rng.poisson(2.0), 0, 12))
        multi_24h = int(np.clip(rng.poisson(5), 0, 25))
        shared_device_iins = int(np.clip(rng.poisson(3.5), 0, 18))
        shared_ip_iins = int(np.clip(rng.poisson(4), 0, 20))

    return {
        "is_fraud": 1,
        "iin_hash": _sha256_short(iin),
        "age": age,
        "gender": gender,
        "region": region,
        "employment_type": employment,
        "monthly_income": income,
        "previous_applications_count": prev_apps,
        "previous_rejections_count": prev_rejections,
        "email_domain": email_domain,
        "phone_age_days": phone_age,
        "phone_type": phone_type,
        "requested_amount": requested_amount,
        "loan_term_days": loan_term,
        "loan_type": loan_type,
        "application_hour": app_hour,
        "application_weekday": app_weekday,
        "device_id_hash": _sha256_short(device_id),
        "device_reuse_count": device_reuse,
        "ip_address_hash": _sha256_short(ip),
        "ip_reuse_count": ip_reuse,
        "proxy_flag": proxy,
        "vpn_flag": vpn,
        "tor_flag": tor,
        "browser_type": browser,
        "os_type": os_type,
        "time_to_fill_application_sec": fill_time,
        "number_of_corrections": corrections,
        "copy_paste_ratio": copy_paste,
        "typing_speed": typing_speed,
        "night_application_flag": night_flag,
        "weekend_flag": weekend_flag,
        "multiple_applications_last_1h": multi_1h,
        "multiple_applications_last_24h": multi_24h,
        "shared_device_with_other_iin_count": shared_device_iins,
        "shared_ip_with_other_iin_count": shared_ip_iins,
    }

def _assign_clusters(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    n = len(df)
    df["cluster_id"] = -1
    df["cluster_size"] = 1
    df["cluster_fraud_ratio"] = 0.0

    fraud_mask = df["is_fraud"] == 1
    fraud_indices = df.index[fraud_mask].tolist()

    if not fraud_indices:
        return df

    n_clusters = max(10, len(fraud_indices) // 25)
    cluster_assignments = rng.integers(0, n_clusters, size=len(fraud_indices))

    for i, idx in enumerate(fraud_indices):
        df.at[idx, "cluster_id"] = int(cluster_assignments[i])

    legit_indices = df.index[~fraud_mask].tolist()
    n_legit_in_cluster = int(len(legit_indices) * 0.02)
    chosen_legit = rng.choice(legit_indices, size=min(n_legit_in_cluster, len(legit_indices)), replace=False)
    for idx in chosen_legit:
        df.at[idx, "cluster_id"] = int(rng.integers(0, n_clusters))

    clustered = df[df["cluster_id"] >= 0]
    for cid in clustered["cluster_id"].unique():
        mask = df["cluster_id"] == cid
        size = int(mask.sum())
        fraud_ratio = round(float(df.loc[mask, "is_fraud"].mean()), 3)
        df.loc[mask, "cluster_size"] = size
        df.loc[mask, "cluster_fraud_ratio"] = fraud_ratio

    unclustered_mask = df["cluster_id"] == -1
    df.loc[unclustered_mask, "cluster_size"] = 1
    df.loc[unclustered_mask, "cluster_fraud_ratio"] = 0.0

    return df

def _compute_statistical_features(df: pd.DataFrame) -> pd.DataFrame:

    df["age_bucket"] = pd.cut(df["age"], bins=[17, 25, 35, 45, 55, 71], labels=False)
    grouped = df.groupby("age_bucket")["monthly_income"]
    df["age_income_zscore"] = grouped.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df["age_income_zscore"] = df["age_income_zscore"].round(3)
    df.drop(columns=["age_bucket"], inplace=True)

    grouped_r = df.groupby("region")["monthly_income"]
    df["region_income_zscore"] = grouped_r.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    df["region_income_zscore"] = df["region_income_zscore"].round(3)

    v = (
        df["multiple_applications_last_1h"] * 2.0
        + df["multiple_applications_last_24h"] * 0.5
        + (1800 - df["time_to_fill_application_sec"].clip(0, 1800)) / 180.0
    )
    v_min, v_max = v.min(), v.max()
    df["application_velocity_score"] = ((v - v_min) / (v_max - v_min + 1e-9) * 9 + 1).round(1)

    signals = (
        df["proxy_flag"] * 1.5
        + df["vpn_flag"] * 1.0
        + df["tor_flag"] * 3.0
        + df["night_application_flag"] * 0.8
        + df["copy_paste_ratio"] * 3.0
        + (df["device_reuse_count"] / 50) * 2.0
        + (df["ip_reuse_count"] / 60) * 2.0
        + (df["shared_device_with_other_iin_count"] / 20) * 2.0
        + (df["shared_ip_with_other_iin_count"] / 25) * 2.0
        + df["cluster_fraud_ratio"] * 3.0
    )
    s_min, s_max = signals.min(), signals.max()
    df["anomaly_score"] = ((signals - s_min) / (s_max - s_min + 1e-9)).round(4)

    return df

def _create_device_pool(rng: np.random.Generator, size: int = 30_000) -> list[str]:
    return [f"device_{i:06d}_{rng.integers(100000, 999999)}" for i in range(size)]

def _create_ip_pool(rng: np.random.Generator, size: int = 25_000) -> list[str]:
    ips = []
    for _ in range(size):
        octets = rng.integers(1, 255, size=4)
        ips.append(f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}")
    return ips

def _create_fraud_device_pool(rng: np.random.Generator, size: int = 200) -> list[str]:
    return [f"fdev_{i:04d}_{rng.integers(100000, 999999)}" for i in range(size)]

def _create_fraud_ip_pool(rng: np.random.Generator, size: int = 150) -> list[str]:
    ips = []
    for _ in range(size):
        octets = rng.integers(1, 255, size=4)
        ips.append(f"{octets[0]}.{octets[1]}.{octets[2]}.{octets[3]}")
    return ips

def generate_dataset(
    n_rows: int = 50_000,
    fraud_rate: float = 0.10,
    seed: int = 42,
) -> pd.DataFrame:
    random.seed(seed)
    rng = np.random.default_rng(seed)

    n_fraud = int(n_rows * fraud_rate)
    n_legit = n_rows - n_fraud

    print(f"Generating {n_rows:,} rows ({n_legit:,} legit, {n_fraud:,} fraud) ...")

    device_pool = _create_device_pool(rng)
    ip_pool = _create_ip_pool(rng)
    fraud_device_pool = _create_fraud_device_pool(rng)
    fraud_ip_pool = _create_fraud_ip_pool(rng)

    rows: list[dict[str, Any]] = []

    print("  Generating legitimate applications ...")
    for i in range(n_legit):
        rows.append(_generate_legit_row(i, rng, device_pool, ip_pool))
        if (i + 1) % 10_000 == 0:
            print(f"    {i + 1:,} / {n_legit:,}")

    print("  Generating fraudulent applications ...")
    for i in range(n_fraud):
        rows.append(_generate_fraud_row(
            i, rng, device_pool, ip_pool, fraud_device_pool, fraud_ip_pool,
        ))
        if (i + 1) % 1_000 == 0:
            print(f"    {i + 1:,} / {n_fraud:,}")

    df = pd.DataFrame(rows)

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)

    print("  Assigning fraud clusters ...")
    df = _assign_clusters(df, rng)

    print("  Computing statistical features ...")
    df = _compute_statistical_features(df)

    column_order = [
        "is_fraud",
        "iin_hash", "age", "gender", "region", "employment_type", "monthly_income",
        "previous_applications_count", "previous_rejections_count",
        "email_domain", "phone_age_days", "phone_type",
        "requested_amount", "loan_term_days", "loan_type",
        "application_hour", "application_weekday",
        "device_id_hash", "device_reuse_count",
        "ip_address_hash", "ip_reuse_count",
        "proxy_flag", "vpn_flag", "tor_flag",
        "browser_type", "os_type",
        "time_to_fill_application_sec", "number_of_corrections",
        "copy_paste_ratio", "typing_speed",
        "night_application_flag", "weekend_flag",
        "multiple_applications_last_1h", "multiple_applications_last_24h",
        "shared_device_with_other_iin_count", "shared_ip_with_other_iin_count",
        "cluster_id", "cluster_size", "cluster_fraud_ratio",
        "age_income_zscore", "region_income_zscore",
        "application_velocity_score", "anomaly_score",
    ]

    df = df[column_order]
    return df

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic fraud detection dataset")
    parser.add_argument("--rows", type=int, default=50_000, help="Total rows to generate")
    parser.add_argument("--fraud-rate", type=float, default=0.10, help="Fraction of fraud rows (0-1)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", type=str, default="data/dataset.csv", help="Output CSV path")
    args = parser.parse_args()

    df = generate_dataset(n_rows=args.rows, fraud_rate=args.fraud_rate, seed=args.seed)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    df.to_csv(args.output, index=False)

    fraud_count = int(df["is_fraud"].sum())
    total = len(df)
    print(f"\nDataset saved to {args.output}")
    print(f"  Total rows:  {total:,}")
    print(f"  Fraud rows:  {fraud_count:,} ({fraud_count / total * 100:.1f}%)")
    print(f"  Legit rows:  {total - fraud_count:,} ({(total - fraud_count) / total * 100:.1f}%)")
    print(f"  Columns:     {len(df.columns)}")
    print(f"  File size:   {os.path.getsize(args.output) / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()

