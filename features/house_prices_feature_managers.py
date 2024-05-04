import os
from pathlib import Path
import pandas as pd

# ▼親ディレクトリの定義
BASE_DIR = str(Path(os.path.abspath('')).parent) + "/"

class HousePricesFeature:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def create_features(self, train_path, test_path):
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)

        train_features = pd.DataFrame()
        test_features = pd.DataFrame()

        # 個々の特徴量クラスのインスタンスを作成し、特徴量を生成
        house_area = HouseArea(self.data_dir)
        train_house_area, test_house_area = house_area.create_feature(train_data, test_data)
        train_features = pd.concat([train_features, train_house_area], axis=1)
        test_features = pd.concat([test_features, test_house_area], axis=1)

        # 他の特徴量クラスも同様に追加
        # ...

        return train_features, test_features

class FeatureBase:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def create_feature(self, train_data, test_data):
        train_feature, test_feature = self.generate_feature(train_data, test_data)
        self.save_feature(train_feature, test_feature)
        return train_feature, test_feature

    def generate_feature(self, train_data, test_data):
        raise NotImplementedError

    def save_feature(self, train_feature, test_feature):
        train_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_train.feather"))
        test_feature.to_feather(os.path.join(self.data_dir, f"{self.__class__.__name__}_test.feather"))

class HouseArea(FeatureBase):
    def generate_feature(self, train_data, test_data):
        train_feature = pd.DataFrame()
        test_feature = pd.DataFrame()

        train_feature['HouseArea'] = train_data['TotalBsmtSF'] + train_data['1stFlrSF'] + train_data['2ndFlrSF']
        test_feature['HouseArea'] = test_data['TotalBsmtSF'] + test_data['1stFlrSF'] + test_data['2ndFlrSF']

        return train_feature, test_feature

# 他の特徴量クラスも同様に実装
# ...

if __name__ == "__main__":
    data_dir = BASE_DIR + "features/feature_data"
    train_path = BASE_DIR + "data/train.csv"
    test_path = BASE_DIR + "data/test.csv"

    house_prices_feature = HousePricesFeature(data_dir)
    train_features, test_features = house_prices_feature.create_features(train_path, test_path)