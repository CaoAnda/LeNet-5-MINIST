# LeNet-5 Pytorch实现
1. 网络卷积核数减少一半
2. 增加dropout层
3. 网络结构适配20x20图像大小

实验结果：

|         | LeNet-5  | LeNet-5+Half Conv | LeNet-5+0.05-Dropout | LeNet-5+0.1-Dropout | LeNet-5+20x20-Input |
| ------- | ------ | --------------- | ------------------ | ----------------- | ----------------- |
| **Acc** | 0.9861 | 0.9776          | 0.9846             | **0.9862**        | 0.9835            |

LeNet-5:

`python main.py`

LeNet-5+Half Conv:

`python main.py --half_conv True`

LeNet-5+0.05-Dropout:

`python main.py --dropout 0.05`

LeNet-5+0.1-Dropout:

`python main.py --dropout 0.1`

LeNet-5+20x20-Input:

`python main.py --input_size 20`