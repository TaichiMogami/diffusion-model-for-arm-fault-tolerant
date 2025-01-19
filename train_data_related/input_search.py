import random


# ランダムな入力を生成
def gen_random_inputs(arm):
    #入力を格納するリストを作成
    res = []
    #アームのばね関節の数だけ繰り返す
    for i in range(arm.spring_joint_count):
        #0から30までのランダムな値をresに追加
        res.append(random.uniform(0, 30))
        res.append(random.uniform(0, 30))
    return res


# x, y: 目標座標
# c: 山登り法の試行回数
# 最もx,yに近かった時の入力、x座標、y座標、角度を返す
def yamanobori(arm, x, y, c):
    #armの初期化
    arm.init()
    #最小の入力を格納するリストを作成
    min_inputs = []
    #最小の距離を10000000に設定
    min_d = 10000000
    #最小のx座標を0に設定
    min_x = 0
    #最小のy座標を0に設定
    min_y = 0
    #最小の角度を0に設定
    min_theta = 0
    #c回繰り返す
    for i in range(c):
        #入力をランダムに生成
        inputs = gen_random_inputs(arm)
        #arm.calcメソッドを呼び出し、引数にinputsを渡す
        arm.calc(inputs.copy())
        #最後のarmのx座標をx_に代入
        x_ = arm.last.x[0][0]
        #最後のarmのy座標をy_に代入
        y_ = arm.last.x[0][1]
        #最後のarmのx座標とy座標から目標座標までの距離を計算
        d = (x-x_)**2+(y-y_)**2
        #もしmin_dがdより大きければ
        if min_d > d:
            #min_dにdを代入
            min_d = d
            #min_inputsにinputsを代入
            min_inputs = inputs
            #min_xにx_を代入
            min_x = x_
            #min_yにy_を代入
            min_y = y_
            #最後のarmの角度をmin_thetaに代入
            min_theta = arm.last.x[1]
    #最小の入力、x座標、y座標、角度を返す
    return min_inputs, min_x, min_y, min_theta
