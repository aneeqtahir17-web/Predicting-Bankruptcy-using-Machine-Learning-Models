# ============================================================
# MACHINE LEARNING: BANKRUPTCY PREDICTION
# Repeated Random Undersampling + Logit + LASSO + Decision Tree
# ============================================================

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("data_cleaned.csv")
target_column = "Bankrupt?"
classification_threshold = 0.3

print("\nDataset shape:", df.shape)
print("\nTarget distribution:")
print(df[target_column].value_counts())
print("\nTarget distribution (%):")
print(df[target_column].value_counts(normalize=True))
print("\nMissing values:", df.isnull().sum().sum())


# ============================================================
# 2. REPEATED RANDOM UNDERSAMPLING SETUP
# ============================================================

bankrupt_df = df[df[target_column] == 1]
non_bankrupt_df = df[df[target_column] == 0]

num_repeats = 10
num_non_bankrupt_sample = 500
test_size = 0.2

decision_tree_param_grid = {
    "max_depth": [3, 4, 5],
    "min_samples_split": [20, 30, 40],
    "min_samples_leaf": [10, 15, 20],
    "criterion": ["gini", "entropy"],
}

repeated_results = []
logit_coefficient_results = []
lasso_coefficient_results = []
ccp_pruning_results = []
roc_curve_outputs = []
best_tree = None
best_tree_feature_names = None
best_tree_run = None
best_tree_f1 = -np.inf
best_tree_ccp_results = None
best_tree_ccp_alpha = None
best_unpruned_tree = None
best_tree_pruning_path = None

if len(non_bankrupt_df) < num_non_bankrupt_sample:
    raise ValueError(
        f"data_cleaned.csv contains only {len(non_bankrupt_df)} non-bankrupt firms; "
        f"{num_non_bankrupt_sample} are required for each sample."
    )

print("\n\n==============================")
print("REPEATED RANDOM UNDERSAMPLING")
print("==============================")
print("Bankrupt observations used each time:", len(bankrupt_df))
print("Random non-bankrupt observations used each time:", num_non_bankrupt_sample)
print("Number of repeats:", num_repeats)


# ============================================================
# 3. TRAIN AND EVALUATE MODELS ON 10 RANDOM SAMPLES
# ============================================================

for run in range(1, num_repeats + 1):
    # Keep every bankrupt firm and draw a new non-bankrupt sample each run.
    sampled_non_bankrupt = non_bankrupt_df.sample(
        n=num_non_bankrupt_sample,
        replace=False,
        random_state=run,
    )
    sampled_df = pd.concat([bankrupt_df, sampled_non_bankrupt], axis=0)
    sampled_df = sampled_df.sample(frac=1, random_state=run).reset_index(drop=True)

    X = sampled_df.drop(columns=[target_column])
    y = sampled_df[target_column]

    # This one stratified split is shared by all three models in a run.
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=run,
        stratify=y,
    )

    # Scaling is needed by the regularized logistic models, but not by trees.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # ---------------- Ordinary logistic regression ----------------
    logit = LogisticRegression(max_iter=5000, solver="lbfgs", random_state=run)
    logit.fit(X_train_scaled, y_train)
    y_prob_logit = logit.predict_proba(X_test_scaled)[:, 1]
    y_pred_logit = (y_prob_logit >= classification_threshold).astype(int)

    logit_coefficient_results.append(pd.DataFrame({
        "Run": run,
        "Feature": ["Intercept", *X.columns],
        "Coefficient": [logit.intercept_[0], *logit.coef_[0]],
    }))

    repeated_results.append({
        "Run": run,
        "Model": "Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred_logit),
        "Precision": precision_score(y_test, y_pred_logit, zero_division=0),
        "Recall": recall_score(y_test, y_pred_logit, zero_division=0),
        "F1-score": f1_score(y_test, y_pred_logit, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob_logit),
        "Best C": np.nan,
        "Best Parameters": np.nan,
    })

    # ---------------- LASSO logistic regression ----------------
    lasso = LogisticRegressionCV(
        penalty="l1",
        solver="saga",
        cv=5,
        scoring="roc_auc",
        max_iter=5000,
        Cs=10,
        random_state=run,
    )
    lasso.fit(X_train_scaled, y_train)
    y_prob_lasso = lasso.predict_proba(X_test_scaled)[:, 1]
    y_pred_lasso = (y_prob_lasso >= classification_threshold).astype(int)
    best_c = lasso.C_[0]

    lasso_coefficients = pd.DataFrame({
        "Run": run,
        "Feature": ["Intercept", *X.columns],
        "Coefficient": [lasso.intercept_[0], *lasso.coef_[0]],
    })
    lasso_coefficient_results.append(lasso_coefficients)

    repeated_results.append({
        "Run": run,
        "Model": "LASSO Logistic Regression",
        "Accuracy": accuracy_score(y_test, y_pred_lasso),
        "Precision": precision_score(y_test, y_pred_lasso, zero_division=0),
        "Recall": recall_score(y_test, y_pred_lasso, zero_division=0),
        "F1-score": f1_score(y_test, y_pred_lasso, zero_division=0),
        "ROC AUC": roc_auc_score(y_test, y_prob_lasso),
        "Best C": best_c,
        "Best Parameters": f"C={best_c:.9f}",
    })

    # Store per-run ROC data so average ROC curves can be produced later.
    model_evaluations = [
        ("Logistic Regression", y_prob_logit),
        ("LASSO Logistic Regression", y_prob_lasso),
    ]

    for model_name, probabilities in model_evaluations:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_test, probabilities)
        roc_curve_outputs.append({
            "Model": model_name,
            "FPR": false_positive_rate,
            "TPR": true_positive_rate,
            "AUC": roc_auc_score(y_test, probabilities),
        })

    # ---------------- Decision tree with 5-fold F1 tuning and CCP pruning ----------------
    # First choose the tree's structural settings.  Cost-complexity pruning is
    # then selected separately from the pruning path of that fitted tree.
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=run)
    tree_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=run),
        param_grid=decision_tree_param_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    tree_grid_search.fit(X_train_raw, y_train)

    # Grow the tree using the selected structural settings, then obtain the
    # candidate ccp_alpha values at which its cost-complexity changes.
    best_tree_params = tree_grid_search.best_params_
    unpruned_tree = DecisionTreeClassifier(random_state=run, **best_tree_params)
    unpruned_tree.fit(X_train_raw, y_train)

    pruning_path = unpruned_tree.cost_complexity_pruning_path(X_train_raw, y_train)
    # The final alpha reduces the tree to its root, so exclude it when another
    # candidate exists.  This avoids selecting a model that cannot classify.
    ccp_alphas = np.unique(pruning_path.ccp_alphas)
    if len(ccp_alphas) > 1:
        ccp_alphas = ccp_alphas[:-1]

    ccp_grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=run, **best_tree_params),
        param_grid={"ccp_alpha": ccp_alphas},
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    ccp_grid_search.fit(X_train_raw, y_train)

    best_ccp_alpha = ccp_grid_search.best_params_["ccp_alpha"]
    decision_tree = DecisionTreeClassifier(
        random_state=run,
        **best_tree_params,
        ccp_alpha=best_ccp_alpha,
    )
    decision_tree.fit(X_train_raw, y_train)
    y_prob_tree = decision_tree.predict_proba(X_test_raw)[:, 1]
    y_pred_tree = (y_prob_tree >= classification_threshold).astype(int)
    tree_f1 = f1_score(y_test, y_pred_tree, zero_division=0)

    # Store the entire pruning path and cross-validation results for audit and
    # for plotting the selected run after repeated sampling is complete.
    ccp_cv_results = pd.DataFrame({
        "Run": run,
        "ccp_alpha": ccp_grid_search.cv_results_["param_ccp_alpha"].astype(float),
        "CV F1 Mean": ccp_grid_search.cv_results_["mean_test_score"],
        "CV F1 Std": ccp_grid_search.cv_results_["std_test_score"],
    })
    ccp_cv_results["Path Impurity"] = ccp_cv_results["ccp_alpha"].map(
        dict(zip(pruning_path.ccp_alphas, pruning_path.impurities))
    )
    ccp_cv_results["Leaves"] = ccp_cv_results["ccp_alpha"].map(
        {
            alpha: DecisionTreeClassifier(
                random_state=run, **best_tree_params, ccp_alpha=alpha
            ).fit(X_train_raw, y_train).get_n_leaves()
            for alpha in ccp_alphas
        }
    )
    ccp_cv_results["Depth"] = ccp_cv_results["ccp_alpha"].map(
        {
            alpha: DecisionTreeClassifier(
                random_state=run, **best_tree_params, ccp_alpha=alpha
            ).fit(X_train_raw, y_train).get_depth()
            for alpha in ccp_alphas
        }
    )
    ccp_cv_results["Selected"] = np.isclose(
        ccp_cv_results["ccp_alpha"], best_ccp_alpha
    )
    ccp_pruning_results.append(ccp_cv_results)

    repeated_results.append({
        "Run": run,
        "Model": "Decision Tree",
        "Accuracy": accuracy_score(y_test, y_pred_tree),
        "Precision": precision_score(y_test, y_pred_tree, zero_division=0),
        "Recall": recall_score(y_test, y_pred_tree, zero_division=0),
        "F1-score": tree_f1,
        "ROC AUC": roc_auc_score(y_test, y_prob_tree),
        "Best C": np.nan,
        "Best Parameters": str({**best_tree_params, "ccp_alpha": best_ccp_alpha}),
    })

    false_positive_rate, true_positive_rate, _ = roc_curve(y_test, y_prob_tree)
    roc_curve_outputs.append({
        "Model": "Decision Tree",
        "FPR": false_positive_rate,
        "TPR": true_positive_rate,
        "AUC": roc_auc_score(y_test, y_prob_tree),
    })

    # Retain the highest test-F1 tree for the requested plot and importances.
    if tree_f1 > best_tree_f1:
        best_tree = decision_tree
        best_tree_feature_names = X.columns.tolist()
        best_tree_run = run
        best_tree_f1 = tree_f1
        best_tree_ccp_results = ccp_cv_results
        best_tree_ccp_alpha = best_ccp_alpha
        best_unpruned_tree = unpruned_tree
        best_tree_pruning_path = pruning_path

    print(
        f"Run {run}: "
        f"Logit F1={repeated_results[-3]['F1-score']:.4f}, "
        f"LASSO F1={repeated_results[-2]['F1-score']:.4f}, "
        f"Tree F1={tree_f1:.4f}, "
        f"Tree parameters={str({**best_tree_params, 'ccp_alpha': best_ccp_alpha})}"
    )


# ============================================================
# 4. SUMMARY TABLES AND MODEL COMPARISON
# ============================================================

repeated_results_df = pd.DataFrame(repeated_results)
repeated_results_df.to_csv("repeated_sampling_results.csv", index=False)

metrics = ["Accuracy", "Precision", "Recall", "F1-score", "ROC AUC"]
summary_metrics = repeated_results_df.groupby("Model")[metrics].agg(["mean", "std"])
summary_metrics.columns = [f"{metric} {stat.title()}" for metric, stat in summary_metrics.columns]
model_comparison = summary_metrics.reset_index()

print("\n\nMODEL COMPARISON")
print(model_comparison.to_string(index=False))

model_comparison.to_csv("model_comparison.csv", index=False)

# One figure: average ROC curves for all three models across the random samples.
fig, axis = plt.subplots(figsize=(8, 6))
model_definitions = [
    ("Logistic Regression", "Logit"),
    ("LASSO Logistic Regression", "LASSO"),
    ("Decision Tree", "Decision Tree"),
]

mean_fpr = np.linspace(0, 1, 100)
for model_name, short_name in model_definitions:
    model_roc_curves = [
        curve for curve in roc_curve_outputs if curve["Model"] == model_name
    ]
    interpolated_tprs = []
    for curve in model_roc_curves:
        interpolated_tpr = np.interp(mean_fpr, curve["FPR"], curve["TPR"])
        interpolated_tpr[0] = 0.0
        interpolated_tpr[-1] = 1.0
        interpolated_tprs.append(interpolated_tpr)
    mean_tpr = np.mean(interpolated_tprs, axis=0)
    mean_auc = np.mean([curve["AUC"] for curve in model_roc_curves])
    axis.plot(mean_fpr, mean_tpr, label=f"{short_name} (mean AUC = {mean_auc:.3f})")

axis.plot([0, 1], [0, 1], "k--", label="Random classifier")
axis.set_xlabel("False Positive Rate")
axis.set_ylabel("True Positive Rate")
axis.set_title("Average ROC Curves Across Random Samples")
axis.legend(loc="lower right")
axis.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("all_models_average_roc_curve.png", dpi=300, bbox_inches="tight")
plt.close(fig)


# ============================================================
# 5. COST-COMPLEXITY PRUNING OUTPUTS
# ============================================================

ccp_pruning_df = pd.concat(ccp_pruning_results, ignore_index=True)
ccp_pruning_df.to_csv("decision_tree_ccp_pruning_results.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(
    best_tree_ccp_results["ccp_alpha"],
    best_tree_ccp_results["CV F1 Mean"],
    marker="o",
)
plt.axvline(best_tree_ccp_alpha, color="tab:red", linestyle="--", label="Selected alpha")
plt.xscale("symlog", linthresh=1e-10)
plt.xlabel("ccp_alpha")
plt.ylabel("5-fold CV F1-score")
plt.title(f"Cost-Complexity Pruning Selection (Run {best_tree_run})")
plt.legend()
plt.tight_layout()
plt.savefig("decision_tree_ccp_selection.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.step(
    best_tree_ccp_results["ccp_alpha"],
    best_tree_ccp_results["Leaves"],
    where="post",
)
plt.xscale("symlog", linthresh=1e-10)
plt.xlabel("ccp_alpha")
plt.ylabel("Number of leaves")
plt.title(f"Tree Size Across the Pruning Path (Run {best_tree_run})")
plt.tight_layout()
plt.savefig("decision_tree_ccp_tree_size.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(best_tree_pruning_path.ccp_alphas, best_tree_pruning_path.impurities, marker="o")
plt.xscale("symlog", linthresh=1e-10)
plt.xlabel("ccp_alpha")
plt.ylabel("Total leaf impurity")
plt.title(f"Leaf Impurity Across the Pruning Path (Run {best_tree_run})")
plt.tight_layout()
plt.savefig("decision_tree_ccp_leaf_impurity.png", dpi=300, bbox_inches="tight")
plt.close()


# ============================================================
# 6. LOGIT/LASSO COEFFICIENT SUMMARIES
# ============================================================

logit_coefficients_df = pd.concat(logit_coefficient_results, ignore_index=True)
logit_coefficient_summary = logit_coefficients_df.groupby("Feature", as_index=False).agg(
    Average_Coefficient=("Coefficient", "mean"),
    Coefficient_Std=("Coefficient", "std"),
)
logit_coefficient_summary["Average_Abs_Coefficient"] = (
    logit_coefficient_summary["Average_Coefficient"].abs()
)
logit_coefficient_summary = logit_coefficient_summary.sort_values(
    "Average_Abs_Coefficient", ascending=False
)
logit_coefficient_summary.to_csv("logit_coefficient_summary.csv", index=False)

lasso_coefficients_df = pd.concat(lasso_coefficient_results, ignore_index=True)
lasso_coefficient_summary = lasso_coefficients_df.groupby("Feature", as_index=False).agg(
    Average_Coefficient=("Coefficient", "mean"),
    Coefficient_Std=("Coefficient", "std"),
    Selection_Count=("Coefficient", lambda values: (values != 0).sum()),
)
lasso_coefficient_summary["Selection_Rate"] = (
    lasso_coefficient_summary["Selection_Count"] / num_repeats
)
lasso_coefficient_summary["Average_Abs_Coefficient"] = (
    lasso_coefficient_summary["Average_Coefficient"].abs()
)
lasso_coefficient_summary = lasso_coefficient_summary.sort_values(
    "Average_Abs_Coefficient", ascending=False
)
lasso_coefficient_summary.to_csv("lasso_coefficient_summary.csv", index=False)

print("\n\nLOGIT COEFFICIENT SUMMARY")
print(logit_coefficient_summary.to_string(index=False))

print("\n\nLASSO COEFFICIENT SUMMARY")
print(lasso_coefficient_summary.to_string(index=False))

old_lasso_best_c_file = Path("lasso_best_c.csv")
if old_lasso_best_c_file.exists():
    old_lasso_best_c_file.unlink()


# ============================================================
# 7. BEST DECISION-TREE PLOTS
# ============================================================

print(f"\n\nBEST DECISION TREE (Run {best_tree_run}, test F1={best_tree_f1:.4f})")

plt.figure(figsize=(28, 14))
plot_tree(
    best_unpruned_tree,
    feature_names=best_tree_feature_names,
    class_names=["Non-bankrupt", "Bankrupt"],
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title(f"Decision Tree Before CCP Pruning (Run {best_tree_run})")
plt.tight_layout()
plt.savefig("best_decision_tree_before_pruning.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(28, 14))
plot_tree(
    best_tree,
    feature_names=best_tree_feature_names,
    class_names=["Non-bankrupt", "Bankrupt"],
    filled=True,
    rounded=True,
    fontsize=7,
)
plt.title(
    f"Best Pruned Decision Tree: Run {best_tree_run} "
    f"(Test F1 = {best_tree_f1:.4f}, ccp_alpha = {best_tree_ccp_alpha:.6g})"
)
plt.tight_layout()
plt.savefig("best_decision_tree.png", dpi=300, bbox_inches="tight")
plt.show()


# ============================================================
# 8. DONE
# ============================================================

print("\nTraining completed successfully.")
print("Files saved:")
print("- repeated_sampling_results.csv")
print("- model_comparison.csv")
print("- all_models_average_roc_curve.png")
print("- logit_coefficient_summary.csv")
print("- lasso_coefficient_summary.csv")
print("- decision_tree_ccp_pruning_results.csv")
print("- decision_tree_ccp_selection.png")
print("- decision_tree_ccp_tree_size.png")
print("- decision_tree_ccp_leaf_impurity.png")
print("- best_decision_tree_before_pruning.png")
print("- best_decision_tree.png")
