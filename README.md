# MINIST_Test
基于minist数据集,利用pytorch学习搭建cnn网络,入门深度学习

# 自定义深度学习代码框架  (deeplearing_base_sructure 分支)
- 功能
    - 参数配置和功能代码分离
    - 保存每次训练的参数,loss,准确率,目前最佳准确率
    - 针对突发情况可以继续训练 -retrain 参数
- data 存放数据集
- output 存放输出,包括log输出和模型参数保存
- module 存放网络结构代码
- config.py 超参数配置,所有参数配置的内容都在这个文件中进行,将参数和模型分离开
- utils.py 常用的基本功能函数,数据集读取,训练,测试
- main.py 主函数,结合不同应用可以改成更具区分度的名字

