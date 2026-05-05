import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
 
from sklearn.model_selection import train_test_split, KFold, cross_val_score, LeaveOneOut
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

print("=" * 60)
print("LOADING DATA")
print("=" * 60)
 
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
 
print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")
 
TARGET = 'Irrigation_Need'
ID_COL = 'id'
 

print("\n" + "=" * 60)
print("FEATURE ENGINEERING")
print("=" * 60)
 
def engineer_features(df):
    df = df.copy()
 
    
    df['Water_Stress'] = df['Temperature_C'] / (df['Soil_Moisture'] + 1e-5)
 
    df['ET_Proxy'] = df['Temperature_C'] * df['Sunlight_Hours'] / (df['Humidity'] + 1e-5)
 
    df['Effective_Rainfall'] = df['Rainfall_mm'] / (df['Wind_Speed_kmh'] + 1e-5)
 
    df['Soil_Health'] = df['Soil_pH'] * df['Organic_Carbon']
 
    df['Moisture_Deficit'] = df['Soil_Moisture'] - df['Rainfall_mm']
 
    df['Salt_Stress'] = df['Electrical_Conductivity'] * df['Soil_pH']
 
    df['Irrigation_Efficiency'] = df['Previous_Irrigation_mm'] / (df['Field_Area_hectare'] + 1e-5)
 
    df['Heat_Index'] = df['Temperature_C'] + 0.33 * df['Humidity'] - 4
 
    df['Rainfall_per_Area'] = df['Rainfall_mm'] / (df['Field_Area_hectare'] + 1e-5)
 
    return df
 
train = engineer_features(train)
test  = engineer_features(test)
 
print(f"Features after engineering: {train.shape[1]}")
 

print("\n" + "=" * 60)
print("ENCODING")
print("=" * 60)
 
cat_cols = train.select_dtypes(include='object').columns.tolist()
cat_cols = [c for c in cat_cols if c != TARGET]
 
print(f"Categorical columns: {cat_cols}")
 
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train[col], test[col]], axis=0).astype(str)
    le.fit(combined)
    train[col] = le.transform(train[col].astype(str))
    test[col]  = le.transform(test[col].astype(str))
    le_dict[col] = le
 

le_y = LabelEncoder()
train[TARGET + '_enc'] = le_y.fit_transform(train[TARGET])
print(f"Target classes: {le_y.classes_}")
print(f"Class distribution:\n{train[TARGET].value_counts()}")
 

drop_cols = [ID_COL, TARGET, TARGET + '_enc']
X = train.drop(columns=drop_cols)
y = train[TARGET + '_enc']
X_test = test.drop(columns=[ID_COL])
 

X_test = X_test[X.columns]
 
print(f"\nFinal feature count: {X.shape[1]}")
print(f"Sample count: {X.shape[0]}")
 

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
 
scaler = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_tr)
X_val_sc = scaler.transform(X_val)
X_test_sc = scaler.transform(X_test)
 

print("\n" + "=" * 60)
print("TRAINING ALL MODELS")
print("=" * 60)
 
models = {
    'Decision Tree':       (DecisionTreeClassifier(max_depth=15, min_samples_leaf=5, random_state=42), False),
    'Naive Bayes':         (GaussianNB(), False),
    'Logistic Regression': (LogisticRegression(max_iter=1000, C=1.0, random_state=42), True),
    'KNN':                 (KNeighborsClassifier(n_neighbors=11), True),
    'Random Forest':       (RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1), False),
    'XGBoost':             (XGBClassifier(n_estimators=500, max_depth=6, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          random_state=42, eval_metric='mlogloss', n_jobs=-1), False),
    'LightGBM':            (LGBMClassifier(n_estimators=500, max_depth=8, learning_rate=0.05,
                                           subsample=0.8, colsample_bytree=0.8,
                                           random_state=42, n_jobs=-1, verbose=-1), False),
}
 
results = {}
for name, (model, use_scale) in models.items():
    Xtr_ = X_tr_sc if use_scale else X_tr
    Xvl_ = X_val_sc if use_scale else X_val
    model.fit(Xtr_, y_tr)
    preds = model.predict(Xvl_)
    acc = accuracy_score(y_val, preds)
    results[name] = acc
    print(f"  {name:25s}: {acc:.4f}")
 


print("\nSaving confusion matrices...")
 
fig, axes = plt.subplots(2, 4, figsize=(22, 10))
axes = axes.flatten()
 
for i, (name, (model, use_scale)) in enumerate(models.items()):
    Xvl_ = X_val_sc if use_scale else X_val
    preds = model.predict(Xvl_)
    cm = confusion_matrix(y_val, preds)
    sns.heatmap(cm, ax=axes[i], annot=True, fmt='d', cmap='Blues',
                xticklabels=le_y.classes_, yticklabels=le_y.classes_)
    axes[i].set_title(f'{name}\nAcc: {results[name]:.4f}', fontsize=10)
    axes[i].set_xlabel('Predicted')
    axes[i].set_ylabel('Actual')
 
axes[-1].axis('off')
plt.suptitle('Confusion Matrices — All Models', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("  Saved: confusion_matrices.png")
 

print("\n" + "=" * 60)
print("5-FOLD CROSS VALIDATION")
print("=" * 60)
 
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
 
for name, (model, use_scale) in models.items():
    X_ = np.array(X_tr_sc if use_scale else X_tr)
    scores = cross_val_score(model, X_, y_tr, cv=kf, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:25s}: {scores.mean():.4f} ± {scores.std():.4f}")
 

print("\n" + "=" * 60)
print("LOOCV (500-sample subset — full dataset too large)")
print("=" * 60)
 
sample_idx = np.random.RandomState(42).choice(len(X_tr), 500, replace=False)
Xs = np.array(X_tr)[sample_idx]
ys = np.array(y_tr)[sample_idx]
 
loo = LeaveOneOut()
for name in ['Decision Tree', 'Naive Bayes']:
    model, use_scale = models[name]
    Xs_ = scaler.transform(Xs) if use_scale else Xs
    loo_sc = cross_val_score(model, Xs_, ys, cv=loo, scoring='accuracy', n_jobs=-1)
    print(f"  {name:25s}: {loo_sc.mean():.4f}")
 

print("\n" + "=" * 60)
print("FINAL MODEL — LightGBM on full training data")
print("=" * 60)
 

from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y)
 
best_model = LGBMClassifier(
    n_estimators=1000,
    max_depth=8,
    learning_rate=0.03,
    num_leaves=63,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
 
print("  Training on full dataset (630k rows)...")
best_model.fit(np.array(X), y, sample_weight=sample_weights)
print("  Done.")
 
# Val accuracy check
val_preds = best_model.predict(np.array(X_val))
print(f"  Val accuracy (final model): {accuracy_score(y_val, val_preds):.4f}")
print(f"\n  Classification report:\n{classification_report(y_val, val_preds, target_names=le_y.classes_)}")
 

feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
feat_imp = feat_imp.sort_values(ascending=False).head(20)
 
plt.figure(figsize=(10, 7))
feat_imp.plot(kind='barh', color='steelblue')
plt.title('Top 20 Feature Importances (LightGBM)', fontsize=13)
plt.xlabel('Importance')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
print("\nSaved: feature_importance.png")
 

print("\n" + "=" * 60)
print("GENERATING SUBMISSION")
print("=" * 60)
 
test_preds_enc = best_model.predict(np.array(X_test))
test_preds     = le_y.inverse_transform(test_preds_enc)
 
submission = pd.DataFrame({
    'id': test[ID_COL],
    'Irrigation_Need': test_preds
})
submission.to_csv('submission.csv', index=False)
 
print(f"  submission.csv saved — {len(submission)} rows")
print(f"  Prediction distribution:\n{submission['Irrigation_Need'].value_counts()}")
 

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Model':<25} {'Val Acc':>10} {'CV Mean':>10} {'CV Std':>10}")
print("-" * 60)
for name in models:
    val_acc = results[name]
    cv_mean = cv_results[name].mean()
    cv_std  = cv_results[name].std()
    print(f"{name:<25} {val_acc:>10.4f} {cv_mean:>10.4f} {cv_std:>10.4f}")
print("=" * 60)
print("Best submission model: LightGBM (full data + class weights)")
print("Output files: submission.csv, confusion_matrices.png, feature_importance.png")
 