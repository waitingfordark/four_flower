# four-flower
![gui-test](./gui-test.png)
这是一个图像识别项目，基于tensorflow，现有的CNN网络可以识别四种花的种类。适合新手对使用tensorflow进行一个完整的图像识别过程有一个大致轮廓。项目包括对数据集的处理，从硬盘读取数据，CNN网络的定义，训练过程，还实现了一个GUI界面用于使用训练好的网络。
## Require
1. Python3.5+
2. tensorflow
3. wxPython
## Quick start
- git clone这个项目
- 导入你喜欢的IDE如pycharm，或者你喜欢的编辑器如Atom。
- 解压input_data.rar到你喜欢的目录。
- 修改train.py中
```
train_dir = 'D:/ML/flower/input_data'  # 训练样本的读入路径
logs_train_dir = 'D:/ML/flower/save'  # logs存储路径
```
为你本机的目录。
- 运行train.py开始训练。
- 训练完成后，修改test.py中的`logs_train_dir = 'D:/ML/flower/save/'`为你的目录。
- 运行test.py或者gui.py查看结果。
