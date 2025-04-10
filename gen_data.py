import os
import random

import pandas as pd
from tqdm import tqdm

from train_data_related import gen_data

# 教師データを生成してdata/train.csvに保存する
if __name__ == "__main__":
    data_list = []
    for i in tqdm(range(100000000)):
        data_list.append(gen_data(random.randint(300, 980), random.randint(100, 600)))

    df = pd.DataFrame(data_list)

    # dataフォルダが存在しない場合、dataフォルダを作成
    if not os.path.exists("data"):
        os.mkdir("data")
    # data/train.csvにデータを保存
    df.to_csv("data/train.csv", index=False)
