from train_data_related import gen_data
import pandas as pd
from tqdm import tqdm
import random
import os

# 教師データを生成してdata/train.csvに保存する
if __name__ == '__main__':
    # データフレームを初期化
    df = pd.DataFrame()
    #100000回繰り返す
    for i in tqdm(range(100000)):
        #gen_dataメソッドを使用して、300から980までのランダムな値と100から600までのランダムな値を引数として渡し、データを生成
        df = pd.concat(
            [df, gen_data(random.randint(300, 980), random.randint(100, 600))])
    #dataフォルダが存在しない場合、dataフォルダを作成
    if not os.path.exists('data'):
        os.mkdir('data')
    #data/train.csvにデータを保存
    df.to_csv('data/train.csv', index=False)
