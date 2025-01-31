import copy

import numpy as np
import pygame

# シミュレーションで使うアームのパーツの実装


class Part:
    # インスタンス生成時の初期化処理
    def __init__(self, pre, base):
        self.pre = pre
        self.base = base
        # x = [[x, y], θ]
        self.x = [[0.0, 0.0], 0.0]
        self.w = 0.0

    # インスタンス生成後に明示的に初期化処理を行う
    def init(self):
        self.x[1] = 0.0
        # 現在のオブジェクトが最後のオブジェクトでない場合、次のオブジェクトのinitメソッドを呼び出す
        if self is not self.base.last:
            self.next.init()

    # 計算
    def calc(self, vals):
        # valsをdeepcopy(深いコピー)したものをvalsに代入
        vals = copy.copy(vals)
        # calced_xメソッドを呼び出し、その結果をself.xに代入
        self.x = self.calced_x(vals)
        # 現在のオブジェクトが最後のオブジェクトでない場合、次のオブジェクトのcalcメソッドを呼び出す
        if self is not self.base.last:
            self.next.calc(vals)

    # 描画
    # display: 画面, colorのデフォルト値は(0, 0, 0)でRGB値を表す
    def draw(self, display, color=(0, 0, 0)):
        # drawpartメソッドを呼び出し、その結果をdisplayとcolorに代入
        self.draw_part(display, color)
        # 現在のオブジェクトが最後のオブジェクトでない場合、次のオブジェクトのdrawメソッドを呼び出す
        if self is not self.base.last:
            self.next.draw(display, color)

    # パーツの先に関節を生やす
    def add_spring_joint(self, c):
        # baseのspring_joint_countに1を加える
        self.base.spring_joint_count += 1
        # 新しいSpringJointクラスのインスタンスを生成し、self.nextに代入
        self.next = SpringJoint(self, self.base, c)
        # baseの最後のオブジェクトにself.nextを代入
        self.base.last = self.next
        # self.nextを返す
        return self.next

    # パーツの先に腕を生やす
    def add_bone(self, length):
        # 新しいBoneクラスのインスタンスを生成し、self.nextに代入
        self.next = Bone(self, self.base, length)
        # baseの最後のオブジェクトにself.nextを代入
        self.base.last = self.next
        # self.nextを返す
        return self.next


# Baseクラスの定義
class Base(Part):
    # インスタンス生成時の初期化処理
    def __init__(self, x_, y_):
        # Partクラスの__init__メソッドの引数preとbaseにselfを渡して呼び出す
        super().__init__(self, self)
        # spring_joint_countを0に初期化
        self.spring_joint_count = 0
        # self.xの初期値を[[x_, y_], 0]に設定
        self.x = [[x_, y_], 0]
        # オブジェクトの最後をselfに設定
        self.last = self

    # xの値を返す関数を定義
    def calced_x(self, vals):
        return self.x

    # 描画する関数を定義
    def draw_part(self, display, color):
        pass


# SpringJointクラスの定義
class SpringJoint(Part):
    # 半径
    r = 10

    # インスタンス生成時の初期化処理
    def __init__(self, pre, base, c):
        # Partクラスの__init__メソッドの引数preとbaseにpreとbaseを渡して呼び出す
        super().__init__(pre, base)
        self.c = c

    # calced_xメソッドを定義
    def calced_x(self, vals):
        # calced_thetaメソッドを呼びだし、引数u1にvalsの0番目、u2にvalsの1番目を渡す
        theta = self.calced_theta(vals[0], vals[1])
        # u1とu2に代入された値を削除
        vals.pop(0), vals.pop(0)
        # 引数preのxの0番目の値及びthetaの値をリスト形式で返す
        return [self.pre.x[0], theta]

    # 描画する関数を定義
    def draw_part(self, display, color):
        # pygame.draw.circleメソッドを使用して、円を描画
        pygame.draw.circle(
            # 円の中心座標をx[0][0], x[0][1]に設定、半径をr(10)に設定
            display,
            color,
            (self.x[0][0], self.x[0][1]),
            self.r,
            5,
        )

    # シミュレーターの角度を計算する関数を定義
    # (u2-u1)/2r + φ(一つ上のジョイントと球体の間の角度) を返す
    def calced_theta(self, u1, u2):
        if isinstance(u1, list):
            u1 = u1[0]
        if isinstance(u2, list):
            u2 = u2[0]
        return (-u1 + u2) / (2 * self.r) + self.pre.x[1]


# Boneクラスの定義
class Bone(Part):
    # __init__メソッドの引数にpre, base, lengthを設定
    def __init__(self, pre, base, length):
        # 長さの設定
        self.length = length
        # Partクラスの__init__メソッドの引数preとbaseにpreとbaseを渡して呼び出す
        super().__init__(pre, base)

    # シミュレーターの角度を計算する関数を定義
    def calced_x(self, vals):
        # xの初期値を[0.0, 0.0]に設定
        x = [0.0, 0.0]
        # thetaの初期値をpreのxの1番目に設定
        theta = self.pre.x[1]
        # xのx座標をpreのx座標から、ジョイントのx座標(長さ×sin(θ))を引いて算出
        x[0] = self.pre.x[0][0] - self.length * np.sin(theta)
        # xのy座標をpreのy座標から、ジョイントのy座標(長さ×cos(θ))を引いて算出
        x[1] = self.pre.x[0][1] - self.length * np.cos(theta)
        # xとthetaをリスト形式で返す
        return [x, theta]

    # 描画する関数を定義
    def draw_part(self, display, color):
        # 一つ上のxから、次のxまでの直線を描画
        pygame.draw.line(
            display,
            color,
            (self.pre.x[0][0], self.pre.x[0][1]),
            (self.x[0][0], self.x[0][1]),
            5,
        )


# class PrintEndEffecta(Part):
#     def __init__(self, pre, base):
#         super().__init__(pre, base)
#     def calced_x(self, vals):
#         print(self.pre.x[0])
#         return self.pre.x
#     def calced_y(self, vals):
#         return self.pre.x[1]
