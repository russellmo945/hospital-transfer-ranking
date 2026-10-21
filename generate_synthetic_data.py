import csv
import random
import math
from datetime import datetime, timedelta
from pathlib import Path

SEED = 42
N_ALL = 3000
N_CTAL = 450

random.seed(SEED)

OUT = Path(__file__).resolve().parent / "data" / "synthetic"
OUT.mkdir(parents=True, exist_ok=True)

services = [
    "Paediatric Medicine",
    "General Surgery",
    "Critical Care Medicine",
    "Cardiology",
    "Neurology",
    "Orthopaedics",
    "Respirology",
    "Haematology/Oncology",
]

regions = [
    ("1:West", "Central West"),
    ("2:Central", "Toronto Central"),
    ("3:East", "Central East"),
    ("4:North", "North Central"),
]

# Fictional hospital names only.
hospitals = [
    ("Community Hospital A", "Central West", 0.82),
    ("Community Hospital B", "Toronto Central", 0.76),
    ("Community Hospital C", "Central East", 0.71),
    ("Community Hospital D", "North Central", 0.66),
    ("Community Hospital E", "Central West", 0.61),
    ("Community Hospital F", "Toronto Central", 0.57),
    ("Community Hospital G", "Central East", 0.53),
    ("Community Hospital H", "North Central", 0.48),
]

transfer_types = ["ACTs", "Voyago", "Repatriation", "Clinical Transfer"]
sexes = ["F", "M"]
fsa_prefixes = ["M4", "M5", "L4", "L5", "L6", "K1", "K2", "N1"]

def sigmoid(x):
    return 1 / (1 + math.exp(-x))

def fy_for_date(d):
    return d.year if d.month >= 4 else d.year - 1

def quarter_for_date(d):
    return f"Q{((d.month - 4) % 12) // 3 + 1}"

def random_datetime(start, end):
    span = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, span))

# -------------------------------------------------------------------
# 1) Generate synthetic ALL_DISCH-like encounter/discharge data
# -------------------------------------------------------------------
all_headers = [
    "PAT_ID","CSN","admissiondtm","InpatientLengthOfStay","adm_from_ed","FSA",
    "Sex","age","fy","ym","direct_adm","disch_disposition","ip_to_other",
    "direct_adm_fr","admit_service","ToHospArea","ToHospName","dischargeDTM",
    "discharge_Service","ED_FRHOSP","IP_FrHosp","FrHospitalName","LHIN_CODE",
    "LHIN_Name","LHIN","HRegion","hRegion"
]

start = datetime(2023, 1, 1, 0, 0)
end = datetime(2025, 3, 31, 23, 59)

all_rows = []

for i in range(N_ALL):
    pat_id = f"SYN-P{100000+i:06d}"
    csn = f"SYN-C{200000+i:06d}"

    admission = random_datetime(start, end - timedelta(days=2))
    los = max(0.2, min(45.0, random.gammavariate(2.0, 2.2)))
    discharge = admission + timedelta(days=los, hours=random.randint(0, 12))

    age = round(random.triangular(0.1, 20.5, 7.0), 2)
    sex = random.choice(sexes)
    hregion, lhin_name = random.choice(regions)
    service = random.choices(
        services,
        weights=[28, 15, 10, 10, 9, 9, 10, 9],
        k=1
    )[0]

    transfer_flag = 1 if random.random() < 0.10 else 0
    to_hosp = random.choice(hospitals)[0] if transfer_flag else ""

    all_rows.append({
        "PAT_ID": pat_id,
        "CSN": csn,
        "admissiondtm": admission.isoformat(sep=" "),
        "InpatientLengthOfStay": round(los, 2),
        "adm_from_ed": random.choice([0, 1]),
        "FSA": random.choice(fsa_prefixes) + str(random.randint(0, 9)),
        "Sex": sex,
        "age": age,
        "fy": fy_for_date(discharge),
        "ym": discharge.strftime("%Y_%m"),
        "direct_adm": random.choice([0, 1]),
        "disch_disposition": "01:Discharged" if not transfer_flag else "02:Transfer to AcuteLTC",
        "ip_to_other": transfer_flag,
        "direct_adm_fr": random.choice(["ED", "Clinic", "Direct", "Other"]),
        "admit_service": service,
        "ToHospArea": lhin_name if transfer_flag else "",
        "ToHospName": to_hosp,
        "dischargeDTM": discharge.isoformat(sep=" "),
        "discharge_Service": service,
        "ED_FRHOSP": random.choice([0, 0, 0, 1]),
        "IP_FrHosp": random.choice([0, 0, 0, 1]),
        "FrHospitalName": random.choice(["", "", "", "Referring Hospital X"]),
        "LHIN_CODE": random.choice(["3505", "3507", "3508", "3510"]),
        "LHIN_Name": lhin_name,
        "LHIN": lhin_name,
        "HRegion": hregion,
        # The modeling notebook currently refers to lowercase hRegion.
        "hRegion": hregion,
    })

all_path = OUT / "synthetic_all_disch.csv"
with all_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=all_headers)
    writer.writeheader()
    writer.writerows(all_rows)

# -------------------------------------------------------------------
# 2) Generate synthetic CTAL-like transfer data
# -------------------------------------------------------------------
ctal_headers = [
    "csn","PAT_ID","transfer_date","dischargeDTM","disch_disposition",
    "First_choice","First_choice_status","Second_choice","Second_choice_status",
    "Patient_Transferred_to_Site","Comments","Type_of_Transfer",
    "TransferCompleted","Reason","Transfer_Site_Region","Region_Site_Name",
    "transfer_outcome","admissiondtm","ToHospName","final_reject","reject",
    "TEMP_date","MONTH","Q","fy","cy","YQ","ym"
]

service_effect = {
    "Paediatric Medicine": 0.20,
    "General Surgery": 0.05,
    "Critical Care Medicine": -0.45,
    "Cardiology": -0.05,
    "Neurology": -0.15,
    "Orthopaedics": 0.10,
    "Respirology": -0.10,
    "Haematology/Oncology": -0.30,
}

type_effect = {
    "ACTs": -0.10,
    "Voyago": 0.05,
    "Repatriation": 0.20,
    "Clinical Transfer": 0.00,
}

hospital_base = {h: p for h, _, p in hospitals}
hospital_region = {h: r for h, r, _ in hospitals}

def accept_probability(enc, hospital, transfer_type, rank):
    base = hospital_base[hospital]
    logit = math.log(base / (1 - base))
    logit += service_effect[enc["discharge_Service"]]
    logit += type_effect[transfer_type]
    logit += -0.25 if rank == 2 else 0.0
    logit += -0.25 if float(enc["InpatientLengthOfStay"]) > 10 else 0.0
    logit += -0.15 if float(enc["age"]) < 1 else 0.0
    logit += 0.10 if enc["LHIN_Name"] == hospital_region[hospital] else -0.05
    return max(0.08, min(0.95, sigmoid(logit)))

selected = random.sample(all_rows, N_CTAL)
ctal_rows = []

for enc in selected:
    first, second = random.sample([h[0] for h in hospitals], 2)
    transfer_type = random.choice(transfer_types)

    accepted1 = random.random() < accept_probability(enc, first, transfer_type, 1)

    has_second = (not accepted1) or (random.random() < 0.35)
    if has_second:
        accepted2 = random.random() < accept_probability(enc, second, transfer_type, 2)
    else:
        accepted2 = False
        second = ""

    first_status = "Accepted" if accepted1 else "Declined"
    second_status = ("Accepted" if accepted2 else "Declined") if has_second else ""

    if accepted1:
        final_site = first
        completed = "Complete"
        transfer_outcome = "1:Yes"
        reason = ""
        final_reject = 0
        reject = 0
    elif accepted2:
        final_site = second
        completed = "Complete"
        transfer_outcome = "1:Yes"
        reason = ""
        final_reject = 0
        reject = 1
    else:
        final_site = ""
        completed = "Incompleted Transfer"
        transfer_outcome = "2:Discharged"
        reason = random.choice([
            "Discharged",
            "Clinical plan changed",
            "No accepting site"
        ])
        final_reject = 1
        reject = 1

    discharge_dt = datetime.fromisoformat(enc["dischargeDTM"])
    transfer_dt = discharge_dt - timedelta(hours=random.randint(2, 36))
    fy = fy_for_date(transfer_dt)
    q = quarter_for_date(transfer_dt)

    ctal_rows.append({
        "csn": enc["CSN"],
        "PAT_ID": enc["PAT_ID"],
        "transfer_date": transfer_dt.date().isoformat(),
        "dischargeDTM": enc["dischargeDTM"],
        "disch_disposition": enc["disch_disposition"],
        "First_choice": first,
        "First_choice_status": first_status,
        "Second_choice": second,
        "Second_choice_status": second_status,
        "Patient_Transferred_to_Site": final_site,
        "Comments": "Synthetic example record; no real patient information.",
        "Type_of_Transfer": transfer_type,
        "TransferCompleted": completed,
        "Reason": reason,
        "Transfer_Site_Region": hospital_region.get(final_site, "") if final_site else "",
        "Region_Site_Name": hospital_region.get(final_site, "") if final_site else "",
        "transfer_outcome": transfer_outcome,
        "admissiondtm": enc["admissiondtm"],
        "ToHospName": final_site,
        "final_reject": final_reject,
        "reject": reject,
        "TEMP_date": transfer_dt.date().isoformat(),
        "MONTH": transfer_dt.month,
        "Q": q,
        "fy": fy,
        "cy": transfer_dt.year,
        "YQ": f"{fy}_{q}",
        "ym": transfer_dt.strftime("%Y_%m"),
    })

ctal_path = OUT / "synthetic_ctal_disch.csv"
with ctal_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=ctal_headers)
    writer.writeheader()
    writer.writerows(ctal_rows)

print(f"Created {len(all_rows):,} rows: {all_path}")
print(f"Created {len(ctal_rows):,} rows: {ctal_path}")

