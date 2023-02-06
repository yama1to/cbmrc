# cbm_rc2

## レポジトリの目的
- 性能評価タスクを用いたCBMRCとESNの比較
- [CBM_RC](https://github.com/katorilab/cbm_rc)の整理

## 目次
- [レポジトリの目的](https://github.com/katorilab/cbm_rc2#レポジトリの目的)
- [cbm_rc2のモデルとタスク](https://github.com/katorilab/cbm_rc2#cbm_rc2のモデルとタスク)
- [フォルダ・ファイルの説明](https://github.com/katorilab/cbm_rc2#フォルダ・ファイルの説明)
- [基本パラメータの説明](https://github.com/katorilab/cbm_rc2#基本パラメータの説明)


## cbm_rc2のモデルとタスク
- モデル
    - CBMRC
    - ESN

- 性能評価タスク
    - ipc
    - memory
    - memory(OU process)
    - narma
    - narma (OU process)
    - santefe
    - speech
    - parity
    - xor

## フォルダ・ファイルの説明
#### explorer/
- 探索アルゴリズムなど

#### generate_dataset/
- 各性能評価タスクのデータ生成を行う。

#### main_モデル_タスク.py
- 実行ファイル（モデルと性能評価タスク）

```
python3 main_モデル_性能評価タスク.py
```

#### test_モデル_タスク.py
- パラメータ最適化・依存性の可視化ファイル（モデルと性能評価タスク)
    - test_all.sh:全てのtest*.pyを実行する。

```
python3 test_モデル_性能評価タスク.py
```
```
sh test_all.sh
```

#### lyon/
- 音声認識のlyonフィルタに関するコード

#### models/
- モデルの定義

#### saved_figures/
- 保存した画像データ<br>
    - delete_figure.sh:全ての保存した画像データを削除する。   
```
sh delete_figure.sh
```

#### saved_models/
- 保存した学習済みモデル
    - delete_models.sh:全ての保存した学習済みモデルを削除する。
 
```
sh delete_models.sh
```

#### tasks/
- 学習データ
- 検証データ
- 評価方法
- プロット

#### trashfigure
- ゴミ箱


## 基本パラメータの説明
|  パラメータ名  |  意味  |
| ---- | ---- |
|  seed  |  乱数生成のためのシード  |
|  NN  |  1サイクルあたりの時間ステップ  |
| NS | 時間ステップ数（サイクル数）  |
| NS0 | 過渡状態の時間ステップ数（学習・評価の際にはこの期間のデータを捨てる） |
| Nu | ノード数（入力） |
| Nh | ノード数（レザバー） |
| Ny | ノード数（出力） |
| Temp | 温度 |
| alpha_i | 結合強度（入力） |
| alpha_r | 結合強度（レザバーのリカレント結合） |
| alpha_b | 結合強度（フィードバック） |
| alpha_s | 結合強度（参照クロック） |
| beta_i | 結合率（入力) |
| beta_r | 結合率（レザバーのリカレント結合） |
| beta_b | 結合率（フィードバック） |
| lambda0 | Ridge回帰の正則化パラメータ | <br>

**Copyright (c) 2022 Katori lab. All Rights Reserved**







# cbmrc
