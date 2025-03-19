import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, StackingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import lightgbm as lgb
import time
import joblib
import json

from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


# =============== 第 0 步：基本设置 ===============
SPLIT_SEED = 42
MODEL_SEED = 23
np.random.seed(MODEL_SEED)

df = pd.read_excel('March 6 预测法补充 data.xlsx') 
X = df.iloc[:, :11]
y = df.iloc[:, 11]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SPLIT_SEED
)

# =============== 第 1 步：先分别优化各基学习器 ===============
# 用 Pipeline 包装 & 用 "model__" 前缀搜索参数

base_param_grids = {
    'rf': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()), 
            ('model', RandomForestRegressor(random_state=MODEL_SEED))
        ]),
        'params': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [None, 5, 10],
            'model__min_samples_split': [2, 5],
            'model__max_features': ['sqrt', 'log2']
        }
    },
    'knn': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            ('model', KNeighborsRegressor())
        ]),
        'params': {
            'model__n_neighbors': [3, 5, 7],
            'model__weights': ['uniform', 'distance'],
            'model__p': [1, 2]
        }
    },
    'xgb': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(objective='reg:squarederror', random_state=MODEL_SEED))
        ]),
        'params': {
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5],
            'model__n_estimators': [100, 200],
            'model__subsample': [0.8, 1.0]
        }
    },
    'lgb': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            ('model', lgb.LGBMRegressor(random_state=MODEL_SEED))
        ]),
        'params': {
            'model__num_leaves': [31, 63],
            'model__learning_rate': [0.01, 0.1],
            'model__n_estimators': [100, 200],
            'model__feature_fraction': [0.8, 1.0]
        }
    },
    'adaboost': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()),
            ('model', AdaBoostRegressor(random_state=MODEL_SEED))
        ]),
        'params': {
            'model__n_estimators': [50, 100],
            'model__learning_rate': [0.1, 0.5, 1.0]
        }
    },
   'mlp': {
        'estimator': Pipeline([
            ('scaler', StandardScaler()), 
            ('model', MLPRegressor(random_state=MODEL_SEED))
        ]),
        'params': {
            # 注意 MLPRegressor 参数需要用 'model__' 前缀指定
            'model__hidden_layer_sizes': [(64,), (128,), (64,32)],
            'model__activation': ['relu', 'tanh'],
            'model__alpha': [1e-4, 1e-3, 1e-2],      # L2 正则化系数
            'model__learning_rate_init': [1e-3, 1e-2],
            'model__solver': ['adam'],
            'model__max_iter': [200, 500]
        }
    }



}

def optimize_model(estimator, param_grid, X, y, model_name):
    """ 执行网格搜索优化，并输出交叉验证结果 """
    print(f"\n=== 优化 {model_name.upper()} ===")
    start = time.time()
    
    gscv = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1,
        verbose=1
    )
    gscv.fit(X, y)
    
    print(f"最佳参数: {gscv.best_params_}")
    print(f"最佳MSE: {-gscv.best_score_:.4f}")
    print(f"耗时: {time.time()-start:.1f}s")
    
    best_estimator = gscv.best_estimator_
    # 再做 5 折交叉验证看看
    cv_scores_mse = cross_val_score(best_estimator, X, y, scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
    cv_scores_r2 = cross_val_score(best_estimator, X, y, scoring='r2', cv=5, n_jobs=-1)
    print(f"5折CV MSE: {-cv_scores_mse.mean():.4f}")
    print(f"5折CV R²:  {cv_scores_r2.mean():.4f}")
    return best_estimator

optimized_models = {}
for name, conf in base_param_grids.items():
    optimized_models[name] = optimize_model(
        conf['estimator'], conf['params'], 
        X_train, y_train, 
        model_name=name
    )



# =============== 第 2 步：对堆叠模型的元学习器做整体搜索 ===============
from sklearn.linear_model import Ridge

# 先构造一个不带 GridSearchCV 的 StackingRegressor
from sklearn.ensemble import StackingRegressor

base_learners = [
    ('rf',       optimized_models['rf']),
    ('knn',      optimized_models['knn']),
    ('xgb',      optimized_models['xgb']),
    ('lgb',      optimized_models['lgb']),
    ('adaboost', optimized_models['adaboost']),
    ('mlp',     optimized_models['mlp']), 
]

# 注意这里 final_estimator 直接是 Ridge()，不再写成 GridSearchCV(...)
stack_model = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(), 
    n_jobs=-1,
    # 设置 StackingRegressor 自身的 CV（默认为 5），
    # 具体多少折可按需求修改
    cv=5  
)

# 我们只搜索元模型 (Ridge) 的超参数
meta_param_grid = {
    'final_estimator__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'final_estimator__solver': ['auto', 'svd', 'cholesky', 'lsqr']
}

# 用 GridSearchCV 搜索整个 "stack_model" 的参数，但实际上只会改动元学习器
stack_gs = GridSearchCV(
    estimator=stack_model,
    param_grid=meta_param_grid,
    scoring='neg_mean_squared_error',
    cv=3,
    n_jobs=-1,
    verbose=1
)

# 进行拟合
stack_gs.fit(X_train, y_train)
print("\n=== 最优元学习器参数 ===")
print(stack_gs.best_params_)

# 在训练集上进行 5折CV 评估
print("\n=== 最优堆叠模型的交叉验证表现 ===")
cv_scores_mse_stack = cross_val_score(stack_gs.best_estimator_, X_train, y_train, 
                                      scoring='neg_mean_squared_error', cv=5, n_jobs=-1)
cv_scores_r2_stack = cross_val_score(stack_gs.best_estimator_, X_train, y_train, 
                                     scoring='r2', cv=5, n_jobs=-1)

print(f"5折CV MSE: {-cv_scores_mse_stack.mean():.4f}")
print(f"5折CV R²:  {cv_scores_r2_stack.mean():.4f}")

# =============== 最终模型在测试集上的表现 ===============
final_stacking_model = stack_gs.best_estimator_
y_pred = final_stacking_model.predict(X_test)
print(f"\n测试集 MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"测试集 R²:  {r2_score(y_test, y_pred):.4f}")

# =============== 基学习器贡献度（元模型系数） ===============
# 此时 final_estimator_ 已经是训练好的 Ridge
# 注意：StackingRegressor 内部会先拟合基模型，然后用 out-of-fold 预测来拟合最终的 Ridge
final_ridge = final_stacking_model.final_estimator_
if hasattr(final_ridge, 'coef_'):
    print("\n基学习器贡献度(系数):")
    for (name, _), coef in zip(final_stacking_model.estimators, final_ridge.coef_):
        print(f"{name.upper():<10} 权重系数: {coef:.4f}")

# =============== （可选）保存模型 ===============
# joblib.dump(final_stacking_model, 'final_stacking_model.pkl')
# with open('optimal_meta_hyperparams.json', 'w') as f:
#     json.dump(stack_gs.best_params_, f, indent=2)
