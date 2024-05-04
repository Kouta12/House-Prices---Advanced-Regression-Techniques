import os
import logging
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# ▼親ディレクトリの定義
BASE_DIR = str(Path(os.path.abspath(''))) + "/"

# ▼ロガーの設定
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ファイルハンドラの設定
file_handler = logging.FileHandler(BASE_DIR + 'logs/train_00.log')
file_handler.setLevel(logging.INFO)

# フォーマッタの設定
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# ロガーにファイルハンドラを追加
logger.addHandler(file_handler)

def load_datasets(feats):
    dfs = [pd.read_feather(f'{BASE_DIR}features/feature_data/{f}_train.feather') for f in feats]
    X_train = pd.concat(dfs, axis=1)
    dfs = [pd.read_feather(f'{BASE_DIR}features/feature_data/{f}_test.feather') for f in feats]
    X_test = pd.concat(dfs, axis=1)
    return X_train, X_test


class HousePricesModel:
    """House Prices コンペティションのモデルを管理するクラス。

    Attributes:
        models (list): 学習済みモデルオブジェクトのリスト。
        target_col (str): 目的変数の列名。
        n_splits (int): クロスバリデーションの分割数。

    Methods:
        train(train_data, target_col, n_splits) -> None:
            学習データを用いて、クロスバリデーションを行いながらモデルを学習する。

        predict(test_data) -> np.ndarray:
            テストデータに対して予測を行う。

        evaluate(test_data, target_col) -> float:
            テストデータに対してモデルを評価し、評価指標を返す。
    """

    def __init__(self, target_col="SalePrice", n_plits=5):
        self.models = []
        self.valid_scores = []
        self.train_logs = []
        self.target_col = target_col
        self.n_splits = n_plits

    def train(self, train_data: pd.DataFrame, params: dict, num_boost_round=10000) -> None:
        """学習データを用いてモデルを学習する。

        Args:
            train_data (pd.DataFrame): 学習データのデータフレーム。
            params (dict): LighGBMのパラメータ
            num_boost_round: 実行回数
        """
        self.params = params
        self.num_boost_round = num_boost_round
        X_train = train_data.drop([self.target_col], axis=1)
        y_train = train_data[self.target_col]

        kf = KFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=0
        )

        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train, y_train)):

            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train.iloc[tr_idx], y_train.iloc[va_idx]

            lgb_train = lgb.Dataset(X_tr, y_tr)
            lgb_eval = lgb.Dataset(X_va, y_va, reference=lgb_train)

            evals_result = {}
            model = lgb.train(
                self.params,
                lgb_train,
                self.num_boost_round,
                valid_sets=[lgb_train, lgb_eval],
                valid_names=["train", "valid"],
                callbacks=[
                    lgb.early_stopping(100),
                    lgb.log_evaluation(500),
                    lgb.record_evaluation(evals_result)
                ]
            )
            y_va_pred = model.predict(X_va, num_iteration=model.best_iteration)
            score = mean_squared_error(y_va, y_va_pred)


            self.models.append(model)
            self.valid_scores.append(score)

            # 学習ログの保存
            for iteration, (train_metric, val_metric) in enumerate(zip(evals_result['train']['l2'], evals_result['valid']['l2']), start=1):
                    logger.info(f"Fold {fold+1}/{self.n_splits} - Iteration: {iteration}, Train MSE: {train_metric}, Validation MSE: {val_metric}")
            logger.info(f"Fold {fold+1}/{self.n_splits} **END**")

            



    def predict(self, test_data: pd.DataFrame) -> None:
        """テストデータに対して予測を行う。

        Args:
            test_data (pd.DataFrame): テストデータのデータフレーム。

        Returns:
            np.ndarray: 予測結果の配列。
        """
        X_test = test_data.drop([self.target_col]) if self.target_col in test_data.columns else test_data
        y_test_pred = np.zeros(X_test.shape[0])
        for model in self.models:
            y_test_pred += model.predict(X_test) / self.n_splits


        return y_test_pred

    def evaluate(self, test_data: pd.DataFrame, target_col: str) -> float:
        """テストデータに対してモデルを評価し、評価指標を返す。

        Args:
            test_data (pd.DataFrame): テストデータのデータフレーム。
            target_col (str): 目的変数の列名。

        Returns:
            float: 評価指標（平均二乗誤差）。
        """
        X_test = test_data.drop([self.target_col])
        y_test = test_data[self.target_col]

        y_test_pred = self.predict(X_test)
        mse = mean_squared_error(y_test, y_test_pred)
        return mse
    

    
if __name__=="__main__":

    # ▼特徴量の指定
    features = {
        "HouseArea",
        "MSSubClass"
    }

    X_train, X_test = load_datasets(features)
    y_train = pd.read_csv(BASE_DIR + "data/train.csv")["SalePrice"]
    train_data = pd.concat([X_train, y_train], axis=1)

    # ▼パラメータの設定
    params = {
        "objective": "regression",
        "metric": "mse",
        "random_seed": 0,
        "learning_rate": 0.02,
        "min_data_in_bin": 3,
        "bagging_freq": 1,
        "bagging_seed": 0,
        "verbose": -1
    }


    # ▼モデルの学習
    model = HousePricesModel()
    model.train(
        train_data,
        params
    )

    # ログの記録
    logger.info("Model trained successfully")