import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from train_data_related import gen_data

# 教師データを生成してdata/train.csvに保存する
if __name__ == "__main__":
    data_list = []
    for i in tqdm(range(10000000)):
        data = gen_data(random.randint(300, 980), random.randint(100, 600))
        data = data.values  # ← ここで1次元データを取り出す！（shapeが(1,15)になる）

        # data_list.append(data[0])  # (1,15) → (15,) にしてappend
        if i % 2 == 0:
            data[0][4] = 0  # 特定次元のデータを固定
            data[0][5] = 30
            # print(f"fixed: {data[0]}")
        data_list.append(data[0])

    data_array = np.array(data_list)  # shape (1000000, 15)
    df = pd.DataFrame(
        data_array, columns=[f"{i}" for i in range(12)] + ["x", "y", "theta"]
    )

    if not os.path.exists("data"):
        os.mkdir("data")
    df.to_csv("data/train.csv", index=False)
