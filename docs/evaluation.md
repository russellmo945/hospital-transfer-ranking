# Evaluation methodology

## Project results

The project analysis covered **1,324 linked discharge and transfer records**, with a tuned XGBoost **Brier score of 0.176** and **38.6% acceptance capture in the top 30% of ranked decisions**. Dividing 0.386 by 0.30 gives approximately **1.29× lift** relative to random ordering. The project also included an Excel patient-flow dashboard; that workbook and confidential source records are not bundled here.

These values summarize the project analysis. The runnable notebooks use generated data and produce their own results; reproducing the project figures requires the corresponding data, split, and fitted model.

## Training and model selection

1. Reserve 20% of encounters using `GroupShuffleSplit` with seed 42.
2. Split the remaining encounters into three `GroupKFold` folds. All hospital decisions for an encounter stay together.
3. Fit imputation, encoding, and any numeric scaling inside each fold's pipeline.
4. Compare logistic regression, histogram gradient boosting, random forest, and XGBoost using mean cross-validation Brier score.
5. Tune XGBoost over depth {2, 4}, learning rate {0.03, 0.1}, and estimator count {150, 300}: eight combinations, each evaluated across three folds.
6. Freeze the selected model using training-fold scores, refit on the training partition, and compute test metrics. The test table compares all candidates but does not choose the winner.

Grouping prevents the same encounter crossing partitions. It does not guarantee patient-disjoint evaluation if a patient has multiple encounters, or generalization to a future time period. For real longitudinal records, choose patient or chronological splits to match the intended use.

## Metrics

| Metric | Meaning |
| --- | --- |
| ROC-AUC | Ability to rank observed accepts above declines across thresholds |
| Average precision | Precision–recall summary for acceptance |
| Brier score | Mean squared error of predicted acceptance probabilities; lower is better |
| Capture at top fraction | Observed accepts among the highest-scored decisions divided by all observed accepts |
| Lift over random | Capture divided by the actual fraction of decisions selected |

The cutoff is `ceil(fraction × number of test decisions)`. Lift uses that realized fraction rather than the nominal percentage to account for rounding in small samples. Equal scores retain row order. The cumulative capture curve starts at (0, 0) and ends at (1, 1) when positives exist.

Capture is pooled across encounters. It does not establish that contacting a fixed number of hospitals per patient will achieve the same success rate, nor does it estimate a causal reduction in outreach workload. Only contacted hospitals have recorded outcomes.

## Candidate ranking and feature timing

The example holds a test encounter's context constant, sets choice rank to 1, and varies the candidate hospital over sites present in the training set. The score is an estimated probability, not a capacity or eligibility decision. Ranking an uncontacted candidate has no observed outcome for comparison.

The available tables contain retrospective discharge information. Final length of stay, discharge service, and discharge timestamps can leak future information for a prospective outreach use case. Real-time evaluation would need elapsed stay, current service, and the request timestamp as known when the decision was made. Current bed availability and staffing are absent.
