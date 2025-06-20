import pandas as pd

from simulator import definition as armdef

from .input_search import yamanobori


# データを生成する関数
def gen_data(x, y):
    # データフレームを初期化
    df = pd.DataFrame()
    # yamanoboriメソッドを使用して、armdef.armとx, y, 100を引数として渡し、input_, x_, y_, thetaを取得
    input_, x_, y_, theta = yamanobori(armdef.arm, x, y, 100)
    df = pd.DataFrame([input_])
    armdef.arm.calc(input_)
    # データフレームにデータを追加
    df["x"] = armdef.arm.last.x[0][0]
    df["y"] = armdef.arm.last.x[0][1]
    df["theta"] = armdef.arm.last.x[1]
    # データフレームを返す
    return df
