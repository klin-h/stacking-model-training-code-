import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
import time
import shap
import joblib
import json

# 设置全局随机种子
RANDOM_STATE = 43
np.random.seed(RANDOM_STATE)

df = pd.read_excel('original_data.xlsx') 
X = df.iloc[:, :11]
y = df.iloc[:, 11]

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

# ---------------------------- 扩展的网格参数 ----------------------------
extended_param_grids = {
    'svr': {
        'estimator': SVR(),
        'params': {
            'kernel': ['rbf', 'linear'],
            'C': [0.1, 1, 10, 100],
            'epsilon': [0.001, 0.01, 0.1]
        }
    },
    'rf': {
        'estimator': RandomForestRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10],
            'max_features': ['sqrt', 'log2']
        }
    },
    'knn': {
        'estimator': KNeighborsRegressor(),
        'params': {
            'n_neighbors': [3, 5, 7, 9],
            'weights': ['uniform', 'distance'],
            'p': [1, 2]
        }
    },
    'xgb': {
        'estimator': XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE),
        'params': {
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'n_estimators': [100, 200],
            'subsample': [0.8, 1.0]
        }
    },
    'lgb': {
        'estimator': lgb.LGBMRegressor(random_state=RANDOM_STATE),
        'params': {
            'num_leaves': [31, 63],
            'learning_rate': [0.01, 0.05, 0.1],
            'n_estimators': [100, 200],
            'feature_fraction': [0.8, 1.0]
        }
    },
    'adaboost': {
        'estimator': AdaBoostRegressor(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0]
        }
    }
}

# ---------------------------- 模型优化 ----------------------------
def optimize_model(estimator, param_grid, X, y, model_name):
    """执行网格搜索优化，并输出交叉验证结果"""
    print(f"\n=== 优化 {model_name.upper()} ===")
    start = time.time()
    
    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1,
        pre_dispatch='2*n_jobs'  # 内存优化
    )
    gscv.fit(X, y)
    
    print(f"最佳参数: {gscv.best_params_}")
    print(f"最佳MSE: {-gscv.best_score_:.4f}")
    print(f"耗时: {time.time()-start:.1f}s")
    
    best_estimator = gscv.best_estimator_
    cv_scores_mse = cross_val_score(best_estimator, X, y, scoring='neg_mean_squared_error', cv=5)
    cv_scores_r2 = cross_val_score(best_estimator, X, y, scoring='r2', cv=5)
    cv_mse_mean, cv_mse_std = -cv_scores_mse.mean(), cv_scores_mse.std()
    cv_r2_mean, cv_r2_std = cv_scores_r2.mean(), cv_scores_r2.std()
    print(f"5折交叉验证MSE: {cv_mse_mean:.4f} ± {cv_mse_std:.4f}")
    print(f"5折交叉验证R²:  {cv_r2_mean:.4f} ± {cv_r2_std:.4f}")
    
    return best_estimator

optimized_models = {}
for name, config in extended_param_grids.items():
    optimized_models[name] = optimize_model(
        config['estimator'], 
        config['params'],
        X_train, y_train,
        model_name=name
    )

# 计算特征重要性（仅适用于树模型）
feature_importance = pd.DataFrame({'Feature': X.columns})
for name in ['rf', 'xgb', 'lgb']:
    feature_importance[name.upper()] = optimized_models[name].feature_importances_

# 计算平均特征重要性
feature_importance['Mean_Importance'] = feature_importance[['RF', 'XGB', 'LGB']].mean(axis=1)
feature_importance = feature_importance.sort_values(by='Mean_Importance', ascending=False)

# ---------------------------- 堆叠模型（含元模型优化） ----------------------------
# 元模型参数优化
meta_param_grid = {
    'alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

# 构建堆叠模型
stacking_model = StackingRegressor(
    estimators=[
        ('svr', optimized_models['svr']),
        ('rf', optimized_models['rf']),
        ('knn', optimized_models['knn']),
        ('xgb', optimized_models['xgb']),
        ('lgb', optimized_models['lgb']),
        ('adaboost', optimized_models['adaboost'])
    ],
    final_estimator=GridSearchCV(
        Ridge(),
        param_grid=meta_param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    ),
    n_jobs=-1
)

# 训练堆叠模型
print("\n=== 训练堆叠模型 ===")
stacking_model.fit(X_train, y_train)

# 提取最佳元模型参数
best_meta_params = stacking_model.final_estimator_.best_params_
print(f"\n最佳元模型参数: {best_meta_params}")

# ====== 额外：堆叠模型的交叉验证结果 ======
cv_scores_mse_stack = cross_val_score(stacking_model, X_train, y_train, scoring='neg_mean_squared_error', cv=5)
cv_scores_r2_stack = cross_val_score(stacking_model, X_train, y_train, scoring='r2', cv=5)
cv_mse_mean_stack, cv_mse_std_stack = -cv_scores_mse_stack.mean(), cv_scores_mse_stack.std()
cv_r2_mean_stack, cv_r2_std_stack = cv_scores_r2_stack.mean(), cv_scores_r2_stack.std()

print(f"\n\033[1;36m=== 堆叠模型5折交叉验证结果 ===\033[0m")
print(f"5折CV MSE: {cv_mse_mean_stack:.4f} ± {cv_mse_std_stack:.4f}")
print(f"5折CV R²:  {cv_r2_mean_stack:.4f} ± {cv_r2_std_stack:.4f}")

print("\n\033[1;36m=== 模型保存 ===\033[0m")
print("模型及参数已保存为：")
print("- optimized_stacking_model.pkl")
print("- optimized_hyperparameters.json")

# 评估模型
y_pred = stacking_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\n测试集MSE: {mse:.4f}")
print(f"测试集R²: {r2:.4f}")

# 计算基学习器的贡献度
print("\n基学习器贡献度:")
coefficients = stacking_model.final_estimator_.best_estimator_.coef_
for name, coef in zip([m[0] for m in stacking_model.estimators], coefficients):
    print(f"{name.upper():<10} 权重系数: {coef:.4f}")

# 输出特征重要性
print("\n=== 特征重要性（基于树模型） ===")
print(feature_importance.to_string(index=False, float_format="%.4f"))
