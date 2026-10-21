# Data and feature guide

The included generator produces **3,000 synthetic encounters** and **450 linked transfer cases** using seed 42. Patient and encounter identifiers begin with `SYN-P` and `SYN-C`. Candidate sites are fictional Community Hospitals A–H. Transfer outcomes are sampled from simulated relationships between hospital, service, transfer type, age, length of stay, and region.

## Input tables

| Table | Fields used for modeling |
| --- | --- |
| `synthetic_all_disch.csv` | `CSN`, `PAT_ID`, `age`, `Sex`, `InpatientLengthOfStay`, `discharge_Service`, `hRegion`, `LHIN_Name` |
| `synthetic_ctal_disch.csv` | `csn`, `PAT_ID`, `dischargeDTM`, `Type_of_Transfer`, `First_choice`, `First_choice_status`, `Second_choice`, `Second_choice_status` |

The exploration notebook also uses `ym`, `ip_to_other`, `HRegion`, and `Patient_Transferred_to_Site`. The generator provides the complete schema for both notebooks.

## Decision records

First- and second-choice records are stacked into one row per observed hospital decision. Exact `Accept`/`Accepted` statuses map to 1; `Decline`/`Declined`/`Reject`/`Rejected` map to 0. Pending, blank, and other statuses are excluded. Candidate hospital names must be nonempty.

Encounter context is joined on both `CSN` and `PAT_ID`. Missing identifiers, duplicate encounter rows, and unmatched decision records raise an error rather than silently creating ambiguous examples. The transfer input expects one row per encounter; event-level source data must first be normalized to that schema.

Predictors are hospital, choice rank, transfer type, age, sex, length-of-stay band, discharge service, region, and discharge hour/day/month. Identifiers, statuses, final destination, and acceptance outcomes are excluded from predictors. Numeric fields receive median imputation; categorical fields receive mode imputation and one-hot encoding with unknown categories handled at prediction time.

## Using other data

Adapt the notebook's `DATA_DIR` and input filenames to local CSV files with the required schema. Validate outcome mappings and event timing before running. Keep confidential files outside the tracked synthetic directory; local `data/private/`, `data/raw/`, model files, and `runs/` outputs are ignored. Do not replace the tracked example CSVs with confidential records.

Synthetic transfer cases are sampled independently of the encounter table's `ip_to_other` flag. Consequently, the exploratory discharge-transfer rate and the transfer-decision cohort describe different generated samples; they should not be reconciled as a single clinical population.
