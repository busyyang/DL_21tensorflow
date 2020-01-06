<center><font size=6>tensorflow知识点</font></center>

## 第一章MNIST
1. tensorflow的占位符是`x=tf.placeholder(tf.float32,[None,10])`前面是数据类型，后面是数据维度,所有的placeholder在训练时候都在`feed_dict`中以字典形式传输数值；
2. 常数`c=tf.constant(0.1,[10,10])`,变量`k=tf.Variable(tf.float32,[10,10])`,参数同样是数据类型和维度；
3. 运算`tf.matmul(a,b)`是乘法；
4. `tf.reduce_sum`和`tf.reduce_mean`是降维求和和求平均，可以指定按照某个轴计算；
5. tensorflow的优化器用法是`train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)`,0.01是学习率；
6. tensorflow在计算值时候需要创建一个会话Session，`sess=tf.Session()`,或者`with tf.Session() as sess:`或`sess = tf.InteractiveSession()`: 
   - tf.InteractiveSession():它能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。这对于工作在交互式环境中的人们来说非常便利，比如使用IPython。
   - tf.Session():需要在启动session之前构建整个计算图，然后启动该计算图。
7. 运行之前需要初始化变量，分配内存`tf.global_variables_initializer().run()`
8. 要计算某个值一个通过`sess.run(k)`的形式计算，也可以`k.eval`;
9. 常用的层都在`tf.nn`中，如`tf.nn.max_pool`或`tf.nn.conv2d`等；
10. dropout的用法`d=tf.nn.dropout(out_last_layer,keep_prob)`;
    
# 第二章CIFAR
1. 创建文件名队列`filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)`,读取可用`reader = tf.WholeFileReader(); key, value = reader.read(filename_queue)`,在读取文件之前需要开始队列`threads = tf.train.start_queue_runners(sess=sess)`,例如：
   ~~~
   # coding:utf-8
   # test.py
    import os
    if not os.path.exists('read'):
        os.makedirs('read/')

    # 导入TensorFlow
    import tensorflow as tf 

    # 新建一个Session
    with tf.Session() as sess:
        # 我们要读三幅图片A.jpg, B.jpg, C.jpg
        filename = ['A.jpg', 'B.jpg', 'C.jpg']
        # string_input_producer会产生一个文件名队列
        filename_queue = tf.train.string_input_producer(filename, shuffle=True, num_epochs=5)
        # reader从文件名队列中读数据。对应的方法是reader.read
        reader = tf.WholeFileReader()
        key, value = reader.read(filename_queue)
        # tf.train.string_input_producer定义了一个epoch变量，要对它进行初始化
        tf.local_variables_initializer().run()
        # 使用start_queue_runners之后，才会开始填充队列
        threads = tf.train.start_queue_runners(sess=sess)
        i = 0
        while True:
            i += 1
            # 获取图片数据并保存
            image_data = sess.run(value)
            with open('read/test_%d.jpg' % i, 'wb') as f:
                f.write(image_data)
    # 程序最后会抛出一个OutOfRangeError，这是epoch跑完，队列关闭的标志

   ~~~
2. 常见图像的数据增广的手段有:
   ~~~
    # Randomly crop a [height, width] section of the image.
    distorted_image = tf.random_crop(reshaped_image, [height, width, 3])

    # Randomly flip the image horizontally.
    distorted_image = tf.image.random_flip_left_right(distorted_image)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    distorted_image = tf.image.random_brightness(distorted_image,max_delta=63)
    distorted_image = tf.image.random_contrast(distorted_image,lower=0.2, upper=1.8)
   ~~~
3. 使用tensorboard需要cd到项目文件夹下，然后`tensorboard --logdir cifar10_train`才行，或者`--logdir`后面跟绝对路径；
4. 设置参数可以通过`FLAGS`,通过`tf.app.run()`获得设定的参数，并用于训练，如果程序入口在main，那么`tf.app.run()`或者`tf.app.run(main)`即可开始训练，如果程序入口为test,那么应该是`tf.app.run(test)`；
    ~~~
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('train_dir', './cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
    ~~~
5. 在使用tensorboard之前需要通过`tf.summary`将参数传输进去，可以定义一个函数进行；
   ~~~
    def _activation_summary(x):
    """Helper to create summaries for activations.

    Creates a summary that provides a histogram of activations.
    Creates a summary that measures the sparsity of activations.

    Args:
        x: Tensor
    Returns:
        nothing
    """
    # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
    # session. This helps the clarity of presentation on tensorboard.
    tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    _activation_summary(conv1)
   ~~~
   将模型中所有可训练参数加入tensorboard：
   ~~~
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
   ~~~
6. 训练过程中打印信息出来，不仅可以通过循环，还可以通过`tf.train.SessionRunHook`来进行，类似与keras中的callback函数;
    ~~~
    class _LoggerHook(tf.train.SessionRunHook):
      """Logs loss and runtime."""

      def begin(self):
        self._step = -1
        self._start_time = time.time()

      def before_run(self, run_context):
        self._step += 1
        return tf.train.SessionRunArgs(loss)  # Asks for loss value.

      def after_run(self, run_context, run_values):
        if self._step % FLAGS.log_frequency == 0:
          current_time = time.time()
          duration = current_time - self._start_time
          self._start_time = current_time

          loss_value = run_values.results
          examples_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
          sec_per_batch = float(duration / FLAGS.log_frequency)

          format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                        'sec/batch)')
          print(format_str % (datetime.now(), self._step, loss_value,
                              examples_per_sec, sec_per_batch))

    with tf.train.MonitoredTrainingSession(
        checkpoint_dir=FLAGS.train_dir,
        hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
               tf.train.NanTensorHook(loss),
               _LoggerHook()],
        config=tf.ConfigProto(
            log_device_placement=FLAGS.log_device_placement)) as mon_sess:
      while not mon_sess.should_stop():
        mon_sess.run(train_op)
    ~~~
7. 多个GPU训练可以参考项目中`cifar10_multi_gpu_train.py`文件，由于没有多个GPU,这个文件暂时没看；

# 第三章目标检测
这章没有用书中的tensorflow Object Detection API,而是用的YOLO.
源码来自于Andrew Ng深度学习项目的作业文件(只有prediction部分，没有train部分)。Keras的YOLO实现可参考https://github.com/qqwweee/keras-yolo3
中文帮助博客：https://blog.csdn.net/weixin_40688204/article/details/89150010