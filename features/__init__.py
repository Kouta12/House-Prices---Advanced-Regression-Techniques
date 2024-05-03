import hashlib
import os
import time
from abc import ABC, abstractmethod
from typing import List, Tuple # 型表示用

import gc # Garbage Collector:自動メモリ管理機能
import pandas as pd


class Feature(ABC):
    """
    機械学習の特徴量生成を抽象化したクラス。

    Attributes:
        data_dir (str): データディレクトリのパス

    Properties:
        name (str): クラス自身の名前

    Methods:
        create_features(train_path, test_path, random_states):
            訓練データとテストデータから特徴量を生成し、
            ファイルパスのタプル (訓練データ用リスト, テストデータ用文字列) を返す。
            random_statesは、(int, pd.Index)のタプルのリストで、
            複数の特徴量セットを生成するための乱数のシードとインデックス。

        categorical_features():
            生成される特徴量のうち、カテゴリ変数となる特徴量のリストを返す。
            クラスメソッド。
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def create_features(self, train_path: str, test_path: str, random_states: List[Tuple[int, pd.Index]]) -> Tuple[List[str], str]:
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def categorical_features():
        raise NotImplementedError
    

class FeatherFeature(Feature):
    """
    特徴量をFeatherファイル形式で保存するための機能を提供するクラス。

    Methods:
        create_features(train_path, test_path, random_states):
            訓練データとテストデータから特徴量を生成し、
            Featherファイルとしてディスク上に保存する。
            既にFeatherファイルが存在する場合は、それを読み込む。
            戻り値は、訓練データ用のFeatherファイルパスのリストと、
            テストデータ用のFeatherファイルパスの文字列からなるタプル。

        get_feature_file(dataset_type, train_path, test_path, random_state):
            特徴量のFeatherファイルパスを生成する。
            dataset_typeは'train'または'test'、random_stateは特徴量セットの識別子。

        get_feature_suffix(train_path, test_path, random_state):
            Featherファイル名のサフィックスを生成する。
            train_pathとtest_pathのハッシュ値とrandom_stateから構成される。

        create_features_impl(train_path, test_path, train_feature_paths, test_feature_path):
            実際の特徴量生成処理を行う抽象メソッド。
            サブクラスでオーバーライドする必要がある。
    """
    def create_features(self, train_path: str, test_path: str, random_states: List[Tuple[int, pd.Index]]) -> Tuple[List[str], str]:
        train_feature_paths = [self.get_feature_file('train', train_path, test_path, random_state=random_state)
                               for (random_state, _) in random_states]
        train_feature_paths_with_index = [(train_feature_paths[i], index) for i, (_, index) in enumerate(random_states)]
        test_feature_path = self.get_feature_file('test', train_path, test_path, 0)

        is_train_cached = all([os.path.exists(train_feature_path) for train_feature_path in train_feature_paths])
        if is_train_cached and os.path.exists(test_feature_path):
            print("There are cache files for feature [{}] (train_path=[{}], test_path=[{}])"
                  .format(self.name, train_path, test_path))
            return train_feature_paths, test_feature_path

        print("Start computing feature [{}] (train_path=[{}], test_path=[{}])".format(self.name, train_path, test_path))
        start_time = time.time()
        self.create_features_impl(train_path=train_path,
                                  test_path=test_path,
                                  train_feature_paths=train_feature_paths_with_index,
                                  test_feature_path=test_feature_path)

        print("Finished computing feature [{}] (train_path=[{}], test_path=[{}]): {:.3} [s]"
              .format(self.name, train_path, test_path, time.time() - start_time))
        return train_feature_paths, test_feature_path
    
    def get_feature_file(self, dataset_type: str, train_path: str, test_path: str, random_state: int) -> str:
        feature_cache_suffix = self.get_feature_suffix(train_path, test_path, random_state)
        filename = self.name + '_' + dataset_type + '_' + feature_cache_suffix + '.feather'
        return os.path.join(self.data_dir, filename)

    @staticmethod
    def get_feature_suffix(train_path: str, test_path, random_state: int) -> str:
        return hashlib.md5(str([train_path, test_path]).encode('utf-8')).hexdigest()[:10] + "_{}".format(random_state)

    @abstractmethod
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        raise NotImplementedError
    

class FeatherFeatureDF(FeatherFeature):
    """
    DataFrameを入力として特徴量を生成するための機能を提供するクラス。

    Methods:
        create_features_impl(train_path, test_path, train_feature_paths, test_feature_path):
            訓練データとテストデータのFeatherファイルを読み込み、
            create_features_from_dataframeメソッドを呼び出して特徴量を生成する。
            生成された特徴量を、指定されたtrain_feature_pathsとtest_feature_pathに
            Featherファイル形式で保存する。

        create_features_from_dataframe(df_train, df_test):
            実際の特徴量生成処理を行う抽象メソッド。
            訓練データ用のDataFrameとテストデータ用のDataFrameを受け取り、
            特徴量として加工したDataFrameのタプル (訓練用, テスト用) を返す。
            サブクラスでオーバーライドする必要がある。
    """
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        df_train = pd.read_feather(train_path)
        df_test = pd.read_feather(test_path)
        train_feature, test_feature = self.create_features_from_dataframe(df_train, df_test)
        for train_feature_path, index in train_feature_paths:
            train_feature.loc[index].reset_index(drop=True).to_feather(train_feature_path)
        test_feature.to_feather(test_feature_path)

    def create_features_from_dataframe(self, df_train: pd.DataFrame, df_test: pd.DataFrame):
        raise NotImplementedError


class FeatherFeaturePath(FeatherFeature):
    """
    ファイルパスを入力として特徴量を生成するための機能を提供するクラス。

    Methods:
        create_features_impl(train_path, test_path, train_feature_paths, test_feature_path):
            訓練データとテストデータのファイルパスを受け取り、
            create_features_from_pathメソッドを呼び出して特徴量を生成する。
            生成された特徴量を、指定されたtrain_feature_pathsとtest_feature_pathに
            Featherファイル形式で保存する。
            処理が終了したら、生成したDataFrameオブジェクトを明示的に削除し、
            ガベージコレクションを実行する。

        create_features_from_path(train_path, test_path):
            実際の特徴量生成処理を行う抽象メソッド。
            訓練データ用のファイルパスとテストデータ用のファイルパスを受け取り、
            特徴量として加工したDataFrameのタプル (訓練用, テスト用) を返す。
            サブクラスでオーバーライドする必要がある。
    """
    def create_features_impl(self, train_path: str, test_path: str, train_feature_paths: List[Tuple[str, pd.Index]],
                             test_feature_path: str):
        train_feature, test_feature = self.create_features_from_path(train_path, test_path)
        for train_feature_path, index in train_feature_paths:
            train_feature.loc[index].reset_index(drop=True).to_feather(train_feature_path)
        test_feature.to_feather(test_feature_path)
        del train_feature, test_feature
        gc.collect()

    def create_features_from_path(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        raise NotImplementedError
