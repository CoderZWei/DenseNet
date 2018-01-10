import tensorflow as tf
from tensorpack import *
from keras.datasets import cifar100
from keras.utils.np_utils import to_categorical
import numpy as np
class DENSE_NET(object):
    def __init__(self,num_blocks,batchsize=1000,height=32,width=32,c_channels=3,growth_rate=4):
        #self.input=input
        self.num_blocks = num_blocks
        self.batchsize=batchsize
        self.height=height
        self.width=width
        self.c_channels=c_channels
        self.growth_rate=growth_rate
        self.first_output_features=growth_rate*2
        self.layers_per_block=4
        self.is_train=True
        self.n_classes=100
        self.weight_decay=1e-4
        self.train_epochs=300
    def conv2d(self,input,out_channels,name='conv2d',strides=[1,1,1,1],kernel_size=3,padding='SAME'):
        with tf.variable_scope(name):
            in_channels=input.get_shape().as_list()[-1]
            kernel=tf.get_variable(name,[kernel_size,kernel_size,in_channels,out_channels])
            output=tf.nn.conv2d(input,kernel,strides,padding)
        return output
    def dropout(self,input):
        if self.is_train:
            output=tf.nn.dropout(input,0.9)
        else:
            output=input
        return output

    #DenseNet-B 加入BN-ReLU-Conv(1×1)-BN-ReLU-Conv(3×3)
    def bottleneck(self,input,out_channels):
        with tf.variable_scope('bottleneck'):
            output=tf.layers.batch_normalization(input,scale=True,training=self.is_train)
            output=tf.nn.relu(output)
            output=self.conv2d(output,out_channels=out_channels*4,kernel_size=1,padding='VALID')
            output=self.dropout(output)
        return output

    def block_unit(self,input,block_index,layers_per_block,growth_rate):
        output=input
        with tf.variable_scope('block%d'%block_index):
            for layer in range(layers_per_block):
                with tf.variable_scope('layer_%d'%layer):
                    output=self.bottleneck(input,out_channels=self.growth_rate)
                    output=self.composite_function(output,out_channels=growth_rate,kernel_size=3)
                    output=tf.concat(axis=3,values=(input,output))
        return output
    def composite_function(self,input,out_channels,kernel_size):
        #Hl function
        with tf.variable_scope('composite_function'):
            output=tf.layers.batch_normalization(input,training=self.is_train)
            output=tf.nn.relu(output)
            output=self.conv2d(output,out_channels=out_channels,kernel_size=kernel_size)
            output=self.dropout(output)
        return output
    def avg_pool(self,input,k):
        ksize=[1,k,k,1]
        strides=[1,k,k,1]
        output=tf.nn.avg_pool(input,ksize,strides,'VALID')
        return output

    #放在两个Dense Block中间，因为每个Dense Block结束后的输出channel个数很多，需要用1*1的卷积核来降维
    #trainsition layer有个参数reduction(范围是0到1)，表示将这些输出缩小到原来的多少倍，默认是0.5，
    # 这样传给下一个Dense Block的时候channel数量就会减少一半，这就是transition layer的作用
    def transition_layer(self,input,reduction=0.5):
        out_channels=int(int(input.get_shape()[-1])*reduction)
        output=self.composite_function(input,out_channels,kernel_size=1)
        #output=tf.nn.avg_pool(output,ksize=2)
        output=self.avg_pool(output,k=2)
        return output
    def transition_layer_classes(self,input):
        #this is last transition to get probabilities by classes
        #BN
        output=tf.layers.batch_normalization(input,training=self.is_train)
        #ReLU
        output=tf.nn.relu(output)
        #average pooling
        last_pool_kernel=int(output.get_shape()[-2])
        output=self.avg_pool(output,k=last_pool_kernel)
        #FC
        features_total = int(output.get_shape()[-1])
        output = tf.reshape(output, [-1, features_total])
        weight=tf.get_variable(shape=[features_total,self.n_classes],name='weight',initializer=tf.random_normal_initializer)
        bias=tf.get_variable(shape=self.n_classes,name='bias',initializer=tf.constant_initializer(0.))
        logits = tf.matmul(output, weight) + bias
        return logits
    def build_net(self,reuse=False):
        self.input = tf.placeholder(tf.float32, shape=[self.batchsize, self.height, self.width, self.c_channels])
        self.labels = tf.placeholder(tf.float32,shape=[self.batchsize,self.n_classes])

        with tf.variable_scope('DenseNet',reuse=reuse):
            x=self.conv2d(self.input,self.first_output_features,'first_conv')
            for block in range(self.num_blocks):
                x=self.block_unit(x,block,self.layers_per_block,self.growth_rate)
                #print('type1:',x)
                if block!=self.num_blocks-1:
                    with tf.variable_scope('transition%d'%block):
                        x=self.transition_layer(x)
                        #print('type2:', x)
            with tf.variable_scope('Transition_to_classes'):
                logits=self.transition_layer_classes(x)
            #print('type',logits)
            self.prediction=tf.nn.softmax(logits)

        #loss
        loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=self.labels))
        l2_loss=tf.add_n([tf.nn.l2_loss(var) for var in tf.trainable_variables()])
        self.loss=loss+l2_loss*self.weight_decay
        self.optimizer=tf.train.AdamOptimizer(0.01).minimize(loss)

        correct_prediction=tf.equal(tf.argmax(self.prediction,1),
                            tf.argmax(self.labels,1))
        self.accuaracy= tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def train(self):
        (X_train,Y_train),(X_test,Y_test)=cifar100.load_data()
        Y_train, Y_test = to_categorical(Y_train, self.n_classes), to_categorical(Y_test, self.n_classes)  # 转换成01序列
        print(np.shape(X_train))
        print(np.shape(Y_train))
        #X_train,X_test=X_train.astype("float32"),X_test.astype("float32")
        #X_train,X_test = X_train.reshape(X_train.shape[0], 32, 32, 3),X_test.reshape(X_test.shape[0], 32, 32, 3)

        nums=int(X_train.shape[0]/self.batchsize)
        #print(Y_test)



        #train_dir = 'data/'
        #data=None

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(self.train_epochs):
                for n in range(nums):
                    predict,label,_,loss=sess.run([self.prediction,self.labels,self.optimizer,self.loss],feed_dict={self.input:X_train[n*self.batchsize:(n+1)*self.batchsize],self.labels:Y_train[n*self.batchsize:(n+1)*self.batchsize]})
                    print('prediction:', np.argmax(predict,1))
                    print('label:',np.argmax(label,1))
                    print('loss：',loss)
                acc=sess.run(self.accuaracy,feed_dict={self.input:X_test[:self.batchsize],self.labels:Y_test[:self.batchsize]})
                print('step',step,acc)


dense_net=DENSE_NET(num_blocks=5)
dense_net.build_net()
dense_net.train()

