from simulator.parts import Base

# 画面の大きさ
width = 1280
height = 720

# 腕の長さ
bone_length = height/10

# バネの硬さ
c = 10

# アームのインスタンスを生成
arm = Base(width/2, height-200)
arm.add_spring_joint(c).add_bone(bone_length)\
   .add_spring_joint(c).add_bone(bone_length)\
   .add_spring_joint(c).add_bone(bone_length)\
   .add_spring_joint(c).add_bone(bone_length)\
   .add_spring_joint(c).add_bone(bone_length)\
   .add_spring_joint(c).add_bone(bone_length)
