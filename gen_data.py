import os
import random

import numpy as np
import pandas as pd
from tqdm import tqdm

from simulator import definition as armdef
from train_data_related import gen_data

# 教師データを生成してdata/train.csvに保存する
if __name__ == "__main__":
    df = pd.DataFrame()
    for i in tqdm(range(100000000)):
        df = pd.concat(
            [df, gen_data(random.randint(300, 780), random.randint(100, 300))]
        )
    # data_array = np.array(data_list)  # shape (1000000, 15)
    # df = pd.DataFrame(
    #     data_array,
    #     columns=[f"{i}" for i in armdef.arm.spring_joint_count * 2]
    #     + ["x", "y", "theta"],
    # )

    if not os.path.exists("data"):
        os.mkdir("data")
    df.to_csv("data/train.csv", index=False)
