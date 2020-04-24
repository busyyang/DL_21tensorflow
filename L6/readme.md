人脸识别网络相应是比较成熟的，其实完全没有必要自己进行训练，基本上来说就是先使用MTCNN进行人脸对齐，在大的图片中将人脸给截取出来，然后使用facenet进行检测，MTCNN与facenet都可以直接在网上下载到模型与weights文件的。可以直接在[davidsandberg/facenet](https://github.com/davidsandberg/facenet)下载到相应的文件。也可以找到21个项目玩转Tensorflow里面查找第六章的代码。

首先使用MTCNN将图片中的人脸截取出来，可以使用[LFW](http://vis-www.cs.umass.edu/lfw/lfw.tgz)文件，将lfw的数据集下载下来解压到raw文件夹下，raw下就是每个人的目录。
~~~bash
python .\src\align\align_dataset_mtcnn.py 
.\datasets\lfw\raw 
.\datasets\lfw\mtcnn_160 
--image_size 160
--margin 32
--gpu_memory_fraction 0.6
~~~
![人脸项目目录](http://m.qpic.cn/psc?/V12DgbRP0a1TXP/PBfbIKZtAJlvfOqE04IdJUF2iEHElaJZY1KlZc2NflZayAjiuiUQpr4iwSkNlbYn30ECULIjwdGVKD7mAEtP9g!!/b&bo=UgLsAAAAAAADB54!&rf=viewer_4)
这样会在`lfw`文件夹下生成mtcnn_160文件夹下生成相应的把人脸剪切出来的图片,最后一个参数gpu_memory_fraction视情况而定，默认是1，如果报`Failed to get convolution algorithm. This is probably because cuDNN failed to initialize, so try looking to see if a warning log message was printed above`这样的错误就加上一个小于1的数，如果还是有就继续改小。

facenet的训练需要大量的图片，在个人电脑上训练是比较困难的事情，所以可以先用别人训练好的模型先看下效果，[下载预训练模型](https://gitlab.fit.cvut.cz/pitakma1/mvi-sp/tree/master/data/20170512-110547)（4个文件一起）放到models目录下，通过`src/validate_on_lfw.py`查看效果：
~~~bash
python src/validate_on_lfw.py
.\datasets\lfw\mtcnn_160
.\models\facenet\20170512-110547
~~~
第一个参数就是图片的路径，第二个参数就是模型文件的路径。可以看到准确率应该在0.99左右。

### 训练
训练需要的数据比较多，可以采用CASIA-Webface数据集，首先还是通过MTCNN将人脸从图片中剪裁出来,这里的image_size稍微大一些，因为在训练时候需要进行一些随机裁剪：
~~~bash
python .\src\align\align_dataset_mtcnn.py 
.\datasets\casia\raw 
.\datasets\casia\mtcnn_182 
--image_size 182
--margin 44
--gpu_memory_fraction 0.6
~~~
由于数据比较多，可能比较慢。
训练的脚本为：
~~~bash
python src/train_softmax.py 
--logs_base_dir logs/facenet/  
--models_base_dir models/myfacenet/   
--data_dir datasets/CASIA-WebFace/mtcnn_182
--image_size 160   
--model_def models.squeezenet  
--lfw_dir datasets/lfw/lfw_mtcnn_160
--optimizer RMSPROP  
--learning_rate -1  
--max_nrof_epochs 80  
--keep_probability 0.8   
--random_crop 
--random_flip   
--learning_rate_schedule_file  data/learning_rate_schedule_classifier_casia.txt  
--weight_decay 5e-5  
--center_loss_factor 1e-2  
--center_loss_alfa 0.9 
--batch_size 64
--gpu_memory_fraction 0.6
~~~
都是一些训练参数，model_def可以有`[squeezenet,inception_resnet_v1,inception_resnet_v2]`三种，自己选择。学习率设置为-1（无效）,并指定文件做阶梯学习率。

### 损失函数
之前一直觉得人脸识别是通过三元损失函数进行训练的，但是由于三元损失完全随机选择不同人的图片的话，有可能已经收敛到比较好的状态，但是识别的结果却不是很好。如果完全按照难例训练，那又难以收敛，所以在训练时候的track要求比较高，所以在实际训练时候，并不是直接使用三元损失函数的。
~~~py
# 三元损失函数tensorflow版本
def triplet_loss(anchor, positive, negative, alpha):
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    with tf.variable_scope('triplet_loss'):
        pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), 1)
        neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), 1)

        basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
        loss = tf.reduce_mean(tf.maximum(basic_loss, 0.0), 0)

    return loss
~~~
中心损失函数是在训练人脸数据集中使用较多的，中心损失就是让每个类别的数据都尽可能聚集到每一类的中心上去，刚开始可以随机指定num_class个中心点，然后通过当前batch计算梯度，更新当前batch的中心点，不能只用中心损失来进行训练，还需要使用softmax的损失，所以损失函数为：
$$L=L_{softmax}+\lambda L_{center}$$

其中，$\lambda$是一个超参数。当$\lambda$越大，结果的每个类别就有比较大的内聚性。
~~~py
# 中心损失函数tensorflow版本
def center_loss(features, label, alfa, nrof_classes):
    """Center loss based on the paper "A Discriminative Feature Learning Approach for Deep Face Recognition"
       (http://ydwen.github.io/papers/WenECCV16.pdf)
    """
    nrof_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [nrof_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alfa) * (centers_batch - features)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(features - centers_batch))
    return loss, centers
~~~
