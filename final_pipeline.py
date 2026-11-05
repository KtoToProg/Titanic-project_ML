import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier

train = pd.read_csv("data/train.csv")
test  = pd.read_csv("data/test.csv")

features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Cabin', 'Embarked']
target = 'Survived'

train_set = train[features + [target]].copy()

plt.figure(figsize=(13, 5))

plt.subplot(1, 2, 1)
survival_by_sex = train_set.groupby('Sex')['Survived'].mean() * 100
plt.bar(['Female', 'Male'], survival_by_sex.values, color=['red', 'blue'])
plt.title('Survival by Sex')
plt.ylabel('Survival %')

plt.subplot(1, 2, 2)
survival_by_pclass = train_set.groupby('Pclass')['Survived'].mean() * 100
plt.bar(['1st', '2nd', '3rd'], survival_by_pclass.values, color=['gold', 'silver', 'brown'])
plt.title('Survival by Class')
plt.ylabel('Survival %')

plt.tight_layout()
plt.show()


def preprocess_fold(raw_train, raw_val=None, raw_test=None):
    def base(df):
        d = df.copy()
        d['Title'] = d['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
        title_map = {'Mr': 'Mr', 'Miss': 'Miss', 'Mrs': 'Mrs', 'Master': 'Master',
                     'Dr': 'Rare', 'Rev': 'Rare', 'Col': 'Rare', 'Major': 'Rare',
                     'Mlle': 'Miss', 'Countess': 'Rare', 'Ms': 'Miss', 'Lady': 'Rare',
                     'Jonkheer': 'Rare', 'Don': 'Rare', 'Dona': 'Rare', 'Mme': 'Mrs',
                     'Capt': 'Rare', 'Sir': 'Rare'}
        d['Title'] = d['Title'].map(title_map).fillna('Rare')
        d['FamilySize'] = d['SibSp'] + d['Parch'] + 1
        d['IsAlone'] = (d['FamilySize'] == 1).astype(int)
        d['HasCabin'] = d['Cabin'].notna().astype(int)
        d['Sex'] = d['Sex'].map({'male': 0, 'female': 1})
        return d

    tr = base(raw_train)
    va = base(raw_val) if raw_val is not None else None
    te = base(raw_test) if raw_test is not None else None

    age_map = tr.groupby(['Title', 'Pclass'])['Age'].median().to_dict()
    age_med = tr['Age'].median()
    fare_med = tr['Fare'].median()
    emb_mode = tr['Embarked'].mode()[0]

    def fill(d):
        if d is None: return None
        d['Age'] = d.apply(lambda r: age_map.get((r['Title'], r['Pclass']), age_med)
        if pd.isna(r['Age']) else r['Age'], axis=1)
        d['Age'] = d['Age'].fillna(age_med)
        d['Fare'] = d['Fare'].fillna(fare_med)
        d['LogFare'] = np.log1p(d['Fare'])
        d['Embarked'] = d['Embarked'].fillna(emb_mode)
        return d

    tr, va, te = fill(tr), fill(va), fill(te)

    dfs = [d for d in [tr, va, te] if d is not None]
    sizes = [len(d) for d in dfs]
    combined = pd.concat(dfs, ignore_index=True)

    for col, prefix in [('Embarked', 'Emb'), ('Title', 'Title')]:
        combined = pd.concat([combined, pd.get_dummies(combined[col], prefix=prefix, dtype=int)], axis=1)
        combined.drop(col, axis=1, inplace=True)

    drop_cols = ['Name', 'Ticket', 'Cabin', 'SibSp', 'Parch', 'Fare', 'Age', 'PassengerId']
    combined = combined.drop([c for c in drop_cols if c in combined.columns], axis=1)

    out, i = [], 0
    for s in sizes:
        out.append(combined.iloc[i:i + s].reset_index(drop=True))
        i += s

    result = []
    idx = 0
    for d in [tr, va, te]:
        if d is None:
            result.append(None)
        else:
            result.append(out[idx])
            idx += 1
    return result[0], result[1], result[2]


print("preprocess_fold (baseline) is ready")
_test = preprocess_fold(train.head(100), train.iloc[100:120], test)
assert _test[1] is not None and not _test[1].isna().any().any(), "NaN in val after preprocessing"
assert list(_test[0].columns) == list(_test[1].columns), "train/val columns mismatch"
print(f"Sanity check OK, {_test[0].shape[1]} features")

y_all = train['Survived'].values
groups_all = train['Name'].str.split(',').str[0].str.strip().values


def make_xgb(seed):
    return XGBClassifier(
        n_estimators=600, max_depth=3, learning_rate=0.02,
        subsample=0.75, colsample_bytree=0.65,
        min_child_weight=5, gamma=0.3,
        reg_alpha=0.5, reg_lambda=3.0,
        random_state=seed, eval_metric='logloss', verbosity=0)


def run_pseudo(global_seed, cv_type='stratified', n_iterations=2, verbose=False):
    if cv_type == 'stratified':
        skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=global_seed)
    else:
        skf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=global_seed)

    test_proba = np.zeros(len(test))
    oof = np.zeros(len(train))

    if cv_type == 'stratified':
        splits = skf.split(train, y_all)
    else:
        splits = skf.split(train, y_all, groups=groups_all)

    for fold, (tr_idx, va_idx) in enumerate(splits):
        raw_tr = train.iloc[tr_idx].reset_index(drop=True)
        raw_va = train.iloc[va_idx].reset_index(drop=True)
        tr_p, va_p, te_p = preprocess_fold(raw_tr, raw_va, test)
        X_tr = tr_p.drop(columns=['Survived'], errors='ignore')
        X_va = va_p.drop(columns=['Survived'], errors='ignore').reindex(columns=X_tr.columns, fill_value=0)
        X_te = te_p.reindex(columns=X_tr.columns, fill_value=0)

        m = make_xgb(global_seed + fold)
        m.fit(X_tr, y_all[tr_idx])
        oof[va_idx] = m.predict_proba(X_va)[:, 1]
        test_proba += m.predict_proba(X_te)[:, 1] / skf.n_splits

    if verbose:
        print(f"  [{cv_type}] Seed {global_seed} iter 0: CV={accuracy_score(y_all, (oof >= 0.5).astype(int)):.4f}")

    for iteration in range(1, n_iterations + 1):
        conf_pos = test_proba > 0.90
        conf_neg = test_proba < 0.10
        conf_mask = conf_pos | conf_neg
        if conf_mask.sum() < 15:
            break

        tr_full, _, te_full = preprocess_fold(train, None, test)
        X_tr_full = tr_full.drop(columns=['Survived'], errors='ignore')
        X_te_full = te_full.reindex(columns=X_tr_full.columns, fill_value=0)

        X_aug = pd.concat([X_tr_full, X_te_full[conf_mask]], ignore_index=True)
        y_aug = np.concatenate([y_all, (test_proba[conf_mask] >= 0.5).astype(int)])

        if cv_type == 'stratified':
            splits_aug = skf.split(X_aug, y_aug)
        else:
            groups_aug = np.concatenate([groups_all, [f'pseudo_{i}' for i in range(conf_mask.sum())]])
            splits_aug = skf.split(X_aug, y_aug, groups=groups_aug)

        test_proba_new = np.zeros(len(test))
        for fold, (tr_idx, va_idx) in enumerate(splits_aug):
            m = make_xgb(global_seed + 100 + fold + iteration * 10)
            m.fit(X_aug.iloc[tr_idx], y_aug[tr_idx])
            test_proba_new += m.predict_proba(X_te_full)[:, 1] / skf.n_splits
        test_proba = test_proba_new
        if verbose:
            print(f"  [{cv_type}] Seed {global_seed} iter {iteration}: confident={conf_mask.sum()}")

    return test_proba, oof


print("\nStratifiedKFold")
seeds = [42, 123, 2024, 7, 99, 555, 314, 2718, 1000, 1337]

results_strat = [run_pseudo(s, cv_type='stratified', verbose=False) for s in seeds]
all_strat = [r[0] for r in results_strat]
all_oofs_strat = [r[1] for r in results_strat]

final_strat = np.mean(all_strat, axis=0)
final_oof_strat = np.mean(all_oofs_strat, axis=0)
print("Stratified CV done")

print("\nGroupKFold")

results_group = [run_pseudo(s, cv_type='group', verbose=False) for s in seeds]
all_group = [r[0] for r in results_group]
all_oofs_group = [r[1] for r in results_group]

final_group = np.mean(all_group, axis=0)
final_oof_group = np.mean(all_oofs_group, axis=0)
print("Group CV done")

final_proba = (final_strat + final_group) / 2
y_pred = (final_proba >= 0.5).astype(int)
print(f"\nFinal blend result: {np.bincount(y_pred)}")

print(f"Stratified only: {np.bincount((final_strat >= 0.5).astype(int))}")
print(f"Group only: {np.bincount((final_group >= 0.5).astype(int))}")

final_oof = (final_oof_strat + final_oof_group) / 2

submission = pd.DataFrame({
    'PassengerId': test['PassengerId'].values,
    'Survived': y_pred
})

# NOTE: OOF metrics are computed using the baseline model (iteration 0) to estimate generalization *prior* to pseudo-labeling.
# Metrics for the final model on OOF are intentionally omitted: since pseudo-labels are derived from the test set,
# any evaluation of the final model on the training data would be biased (data leakage).
# The final model is evaluated using the Public LB as a secondary signal.

oof_preds = (final_oof >= 0.5).astype(int)

print("\nFinal OOF Metrics (Blend Strat+Group):")
print(f"Accuracy:  {accuracy_score(y_all, oof_preds):.5f}")
print(f"Precision: {precision_score(y_all, oof_preds):.5f}")
print(f"Recall:    {recall_score(y_all, oof_preds):.5f}")
print(f"F1-score:  {f1_score(y_all, oof_preds):.5f}")
print(f"ROC-AUC:   {roc_auc_score(y_all, final_oof):.5f}")

submission.to_csv('submission.csv', index=False)

sub_check = pd.read_csv('submission.csv')
print(f"\nRows: {len(sub_check)} | Unique IDs: {sub_check['PassengerId'].nunique()}")
print(f"Min/Max: {sub_check['PassengerId'].min()}/{sub_check['PassengerId'].max()}")
print(f"Class distribution: {sub_check['Survived'].value_counts().to_dict()}")