"""Decision-table construction and pooled outreach-ranking metrics."""

import numpy as np
import pandas as pd


FEATURES = [
    "Hospital", "choice_rank", "Type_of_Transfer", "age", "Sex", "LOS_cat",
    "discharge_Service", "hRegion", "LHIN_Name", "discharge_hour",
    "discharge_dow", "discharge_month",
]


def status_to_label(value):
    """Only explicit decisions become labels; pending/negated text stays unknown."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().casefold()
    if text in {"accept", "accepted"}:
        return 1
    if text in {"decline", "declined", "reject", "rejected"}:
        return 0
    return np.nan


def build_decisions(encounters, transfers):
    """Join on encounter AND patient identity, rejecting ambiguous or missing joins."""
    transfers = transfers.rename(columns={"csn": "CSN"})
    context_cols = ["CSN", "PAT_ID", "age", "Sex", "InpatientLengthOfStay",
                    "discharge_Service", "hRegion", "LHIN_Name"]
    required = ["CSN", "PAT_ID", "dischargeDTM", "Type_of_Transfer",
                "First_choice", "First_choice_status", "Second_choice", "Second_choice_status"]
    for frame, columns in [(encounters, context_cols), (transfers, required)]:
        missing = set(columns) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")
        if frame[["CSN", "PAT_ID"]].isna().any().any():
            raise ValueError("Encounter and patient identifiers must not be missing")
    if encounters["CSN"].duplicated().any():
        raise ValueError("Encounter context must have one row per CSN")
    if transfers["CSN"].duplicated().any():
        raise ValueError("Transfer table must have one row per CSN")
    parts = []
    for prefix, rank in [("First", 1), ("Second", 2)]:
        part = transfers[required].copy()
        part["Hospital"] = part[f"{prefix}_choice"].fillna("").astype(str).str.strip()
        part["Status"] = part[f"{prefix}_choice_status"]
        part["choice_rank"] = rank
        part["y_accept"] = part["Status"].map(status_to_label)
        parts.append(part[part.Hospital.ne("") & part.y_accept.notna()])
    decisions = pd.concat(parts, ignore_index=True)
    if decisions.empty:
        raise ValueError("No explicit accepted or declined decisions")
    decisions["y_accept"] = decisions.y_accept.astype(int)
    decisions = decisions.merge(encounters[context_cols], on=["CSN", "PAT_ID"],
                                how="left", validate="many_to_one", indicator=True)
    if decisions["_merge"].ne("both").any():
        raise ValueError("Every decision must match its encounter and patient context")
    decisions = decisions.drop(columns="_merge")
    dt = pd.to_datetime(decisions["dischargeDTM"], format="mixed", errors="coerce")
    decisions["discharge_hour"] = dt.dt.hour
    decisions["discharge_dow"] = dt.dt.dayofweek
    decisions["discharge_month"] = dt.dt.month
    decisions["LOS_cat"] = pd.cut(
        pd.to_numeric(decisions["InpatientLengthOfStay"], errors="coerce"),
        bins=[-0.1, 1, 2, 4, 10, np.inf], labels=["<=1", "2", "3-4", "5-10", "10+"],
    )
    return decisions


def topk_capture(y_true, y_score, fraction):
    """Fraction of all positives captured in the top ceil(fraction * n) decisions."""
    labels = np.asarray(y_true, dtype=float)
    scores = np.asarray(y_score, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels):
        raise ValueError("Labels and scores must be nonempty matching vectors")
    if not 0 < fraction <= 1 or not np.isfinite(scores).all():
        raise ValueError("Fraction must be in (0, 1] and scores must be finite")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be binary")
    cutoff = int(np.ceil(fraction * len(labels)))
    order = np.argsort(-scores, kind="stable")[:cutoff]
    return float(labels[order].sum() / labels.sum()) if labels.sum() else np.nan


def lift_over_random(y_true, y_score, fraction):
    """Use the actual selected fraction so finite-sample rounding is accounted for."""
    capture = topk_capture(y_true, y_score, fraction)
    actual_fraction = int(np.ceil(fraction * len(y_true))) / len(y_true)
    return capture / actual_fraction
