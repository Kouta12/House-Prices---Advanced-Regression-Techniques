import os
import csv
import time
from pathlib import Path
from contextlib import contextmanager

import pandas as pd

# ▼親ディレクトリの定義
BASE_DIR = str(Path(os.path.abspath('')).parent)

# ▼特徴量メモCSVファイル作成
def create_memo(col_name: str, desc: str):
    file_path = BASE_DIR + "/features/_features_memo.csv"
    if not os.path.isfile(file_path):
        with open(file_path, "w"): pass

    with open(file_path, "r+") as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

        # 書き込もうとしている特徴量がすでに書き込まれていないかチェック
        col = [line for line in lines if line.split(",")[0] == col_name]
        if len(col) != 0:
            return
        
        writer = csv.writer(f)
        writer.writerow([col_name, desc])

# ▼タイマー
@contextmanager
def timer(name):
    t0 = time.time()
    print(f"[{name}] start")
    yield
    print(f"[{name}] done in {time.time() - t0:.0f} s")


class HousePricesFeature:
    """House Prices コンペティションの特徴量を管理するクラス。

    Args:
        data_dir (str): 特徴量を保存するディレクトリのパス。

    Attributes:
        data_dir (str): 特徴量を保存するディレクトリのパス。

    Methods:
        create_features(train_path, test_path) -> Tuple[pd.DataFrame, pd.DataFrame]:
            学習データとテストデータから特徴量を生成し、特徴量のデータフレームを返す。
    """

    def __init__(self, data_dir):
        """コンストラクタ。
        Args:
            data_dir (str): 特徴量を保存するディレクトリのパス。
        """
        self.data_dir = data_dir

    def create_features(self, train_path, test_path):
        """学習データとテストデータから特徴量を生成し、特徴量のデータフレームを返す。

        Args:
            train_path (str): 学習データのファイルパス。
            test_path (str): テストデータのファイルパス。
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: 
                生成された特徴量のデータフレーム (train_features, test_features)。
                - train_features (pd.DataFrame): 学習データの特徴量のデータフレーム。
                - test_features (pd.DataFrame): テストデータの特徴量のデータフレーム。
        """
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_features = pd.DataFrame()
        test_features = pd.DataFrame()

        # ▼個々の特徴量クラスのインスタンスを作成し、特徴量を生成
        house_area = HouseArea(self.data_dir)
        train_house_area, test_house_area = house_area.create_feature(train_data, test_data)
        train_features = pd.concat([train_features, train_house_area], axis=1)
        test_features = pd.concat([test_features, test_house_area], axis=1)

        ms_sub_class = MSSubClass(self.data_dir)
        train_ms_sub_class, test_ms_sub_class = ms_sub_class.create_feature(train_data, test_data)
        train_features = pd.concat([train_features, train_ms_sub_class], axis=1)
        test_features = pd.concat([test_features, test_ms_sub_class], axis=1)

        # 他の特徴量クラスも同様に追加
        # ...

        return train_features, test_features

class FeatureBase:
    """特徴量生成の基底クラス。

    個々の特徴量クラスはこのクラスを継承し、generate_feature メソッドを実装する。

    Args:
        data_dir (str): 特徴量を保存するディレクトリのパス。

    Attributes:
        data_dir (str): 特徴量を保存するディレクトリのパス。

    Methods:
        create_feature(train_data, test_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
            学習データとテストデータから特徴量を生成し、特徴量のデータフレームを返す。
            生成された特徴量は、指定されたディレクトリに Feather 形式で保存される。

        generate_feature(train_data, test_data) -> Tuple[pd.DataFrame, pd.DataFrame]:
            学習データとテストデータから特徴量を生成する抽象メソッド。
            サブクラスでこのメソッドを実装する必要がある。

        save_feature(train_feature, test_feature) -> None:
            生成された特徴量を Feather 形式で保存する。
    """
    def __init__(self, data_dir):
        """コンストラクタ。

        Args:
            data_dir (str): 特徴量を保存するディレクトリのパス。
        """
        self.data_dir = data_dir

    def create_feature(self, train_data, test_data):
        """学習データとテストデータから特徴量を生成し、特徴量のデータフレームを返す。
        生成された特徴量は、指定されたディレクトリに Feather 形式で保存される。

        Args:
            train_data (pd.DataFrame): 学習データのデータフレーム。
            test_data (pd.DataFrame): テストデータのデータフレーム。
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                生成された特徴量のデータフレーム (train_feature, test_feature)。
                - train_feature (pd.DataFrame): 学習データの特徴量のデータフレーム。
                - test_feature (pd.DataFrame): テストデータの特徴量のデータフレーム。
        """
        with timer(name=self.__class__.__name__):
            train_feature, test_feature = self.generate_feature(train_data, test_data)
            self.save_feature(train_feature, test_feature)
        return train_feature, test_feature

    def generate_feature(self, train_data, test_data):
        """学習データとテストデータから特徴量を生成する抽象メソッド。
        サブクラスでこのメソッドを実装する必要がある。

        Args:
            train_data (pd.DataFrame): 学習データのデータフレーム。
            test_data (pd.DataFrame): テストデータのデータフレーム。
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                生成された特徴量のデータフレーム (train_feature, test_feature)。
                - train_feature (pd.DataFrame): 学習データの特徴量のデータフレーム。
                - test_feature (pd.DataFrame): テストデータの特徴量のデータフレーム。
        """
        raise NotImplementedError

    def save_feature(self, train_feature, test_feature):
        """生成された特徴量を Feather 形式で保存する。

        Args:
            train_feature (pd.DataFrame): 学習データの特徴量のデータフレーム。
            test_feature (pd.DataFrame): テストデータの特徴量のデータフレーム。
        """
        train_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_train.feather"))
        test_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_test.feather"))


# ▼特徴量作成
class HouseArea(FeatureBase):
    def generate_feature(self, train_data, test_data):
        train_feature = pd.DataFrame()
        test_feature = pd.DataFrame()

        train_feature['HouseArea'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
        test_feature['HouseArea'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']

        create_memo("HouseArea", "家の総面積: 地下の面積＋1階の面積＋2階の面積")
        return train_feature, test_feature
    
class MSSubClass(FeatureBase):
    def generate_feature(self, train_data, test_data):
        train_feature = pd.DataFrame()
        test_feature = pd.DataFrame()

        train_feature["MSSubClass"] = train_data["MSSubClass"]
        test_feature["MSSubClass"] = test_data["MSSubClass"]
        create_memo("MSSubClass", "売却の対象となる住居タイプ")
        return train_feature, test_feature

        

# 他の特徴量クラスも同様に実装
# ...

if __name__ == "__main__":
    data_dir = BASE_DIR + "/features/feature_data"
    train_path = BASE_DIR + "/data/train.csv"
    test_path = BASE_DIR + "/data/test.csv"

    house_prices_feature = HousePricesFeature(data_dir)
    train_features, test_features = house_prices_feature.create_features(train_path, test_path)