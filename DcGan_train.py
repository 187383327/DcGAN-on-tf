import tensorflow as tf
import Config
import reader
import numpy as np


def generator(z,g_depths,is_training):
    with tf.variable_scope('g'):
        net = tf.layers.dense(z,4*4*g_depths[0])
        net = tf.reshape(net,(-1,4,4,g_depths[0]))
        net = tf.layers.batch_normalization(net,training=is_training)
        net = tf.nn.relu(net)

        net = tf.layers.conv2d_transpose(net,g_depths[1],[5,5],strides=[2,2],padding='SAME')
        net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training))

        net = tf.layers.conv2d_transpose(net,g_depths[2],[5,5],strides= [2,2], padding='SAME' )
        net = tf.nn.relu(tf.layers.batch_normalization(net,training=is_training))

        net = tf.layers.conv2d_transpose(net,g_depths[3], [5,5], strides=[2,2], padding='SAME')
        net = tf.nn.relu(tf.layers.batch_normalization(net, training=is_training))

        net = tf.layers.conv2d_transpose(net,g_depths[4], [5,5], strides=[2,2], padding='SAME')
        # output
        net = tf.tanh(net)
        return net

def discriminator(x,d_depths,is_training):
    with tf.variable_scope('d',reuse=tf.AUTO_REUSE):
        net = tf.layers.conv2d(x,d_depths[1],[5,5],[2,2],padding='SAME')
        net = tf.nn.leaky_relu(tf.layers.batch_normalization(net,training=is_training),alpha=0.2)

        net = tf.layers.conv2d(net,d_depths[2],[5,5],[2,2],padding='SAME')
        net = tf.nn.leaky_relu(tf.layers.batch_normalization(net,training=is_training),alpha=0.2)

        net = tf.layers.conv2d(net,d_depths[3],[5,5],[2,2],padding='SAME')
        net = tf.nn.leaky_relu(tf.layers.batch_normalization(net,training=is_training),alpha=0.2)

        net = tf.layers.conv2d(net,d_depths[4],[5,5],[2,2],padding='SAME')
        net = tf.nn.leaky_relu(tf.layers.batch_normalization(net,training=is_training))

        #classifier
        net = tf.layers.flatten(net)
        net = tf.layers.dense(net,1)
        return net

def bulid_model(config,z,x):
    gz = generator(z,config.g_depths,config.is_training)
    dx = discriminator(x,config.d_depths,config.is_training)
    dgz = discriminator(gz,config.d_depths,config.is_training)

    var_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='g')
    var_d = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope='d')

    # loss_g = -tf.reduce_mean(tf.log(dgz))
    # loss_d = -tf.reduce_mean(tf.log(1-dgz))-tf.reduce_mean(tf.log(dx))
    loss_g = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgz,labels=tf.ones_like(dgz)))
    loss_d = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dx,labels=tf.ones_like(dx)))\
                    + tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dgz,labels=tf.zeros_like(dgz)))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        g_slover = tf.train.AdamOptimizer(config.lr,beta1=config.momentum).minimize(loss_g,var_list=var_g)
        d_slover = tf.train.AdamOptimizer(config.lr,beta1=config.momentum).minimize(loss_d,var_list=var_d)

    return gz, dgz,loss_g,loss_d,g_slover,d_slover


def train():
    config = Config.Config()
    dataset = r'F:\dataSet\faces'
    img = reader.get_image(dataset,config.batch_size)

    z = tf.placeholder(tf.float32,shape=[None,100])
    x = tf.placeholder(tf.float32,shape=[None,64,64,3])
    gz, dgz, loss_g, loss_d, g_slover, d_slover = bulid_model(config,z,x)
    tf.summary.image('g',gz,9)
    tf.summary.scalar('dgz',tf.reduce_mean(tf.sigmoid(dgz)))
    merge_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter('./logs',tf.get_default_graph())
    saver = tf.train.Saver(max_to_keep=2)
    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        if tf.train.latest_checkpoint('./model'):
            saver.restore(tf.train.latest_checkpoint('./model'))
        else :
            sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        i = 0
        try:
            # while not coord.should_stop():
            while True:
                train_data = sess.run(img)
                z_data = np.random.uniform(-1,1,[config.batch_size,100])
                _, lossd, = sess.run([d_slover,loss_d],feed_dict={z:z_data,x:train_data})
                _, lossg = sess.run([g_slover,loss_g],feed_dict={z:z_data})
                i +=1

                if i%50==0:
                    print('loss_d is ',lossd)
                    print('loss_g is ',lossg)
                    summary = sess.run(merge_summary,feed_dict={z:z_data})
                    writer.add_summary(summary,global_step=i)

        except tf.errors.OutOfRangeError:
            print('train done')
            saver.save(sess,'./model',i)
            coord.request_stop()
        coord.join(threads)

if __name__=='__main__':
    train()