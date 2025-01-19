from simulator import definition as armdef
from .input_search import yamanobori
import pandas as pd

# データを生成する関数
def gen_data(x, y):
    # データフレームを初期化
    df = pd.DataFrame()
    # yamanoboriメソッドを使用して、armdef.armとx, y, 100を引数として渡し、input_, x_, y_, thetaを取得
    input_, x_, y_, theta = yamanobori(armdef.arm, x, y, 100)
    # データフレームにデータを追加
    df = pd.DataFrame([input_])
    # データフレームにデータを追加
    df['x'] = x_
    df['y'] = y_
    df['theta'] = theta
    # データフレームを返す
    return df
