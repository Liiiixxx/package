# 整体
* dataset 数据预处理，包括张量转换和标签转换
* main 主函数， 其中预测部分转到test.py中进行
* unet 网络结构
# 说明
* 网络最后一层没有softmax，训练中使用交叉熵损失函数，torch.nn.CrossEntropyLoss()
 torch.nn.CrossEntropyLoss()函数输入不要求onehot编码模式，所以在dataset中将rgb特征转成了数字标签；
  
