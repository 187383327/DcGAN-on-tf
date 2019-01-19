import tensorflow as tf
import os

def get_image(dataset_path,batch_size):
    img_list = [os.path.join(dataset_path, i)  for i in os.listdir(dataset_path)]
    img_queue = tf.train.string_input_producer(img_list,num_epochs=50,shuffle=True)
    reader = tf.WholeFileReader()
    _, img_bytes = reader.read(img_queue)
    img = tf.image.decode_jpeg(img_bytes)
    img.set_shape((96,96,3))
    img = tf.image.central_crop(img,0.8)
    img = tf.image.resize_images(img,size=[64,64])
    img = tf.cast(tf.divide(img,255.0),tf.float32)
    img = (img - 0.5)*2
    return tf.train.batch([img],batch_size=batch_size,num_threads=4,capacity=batch_size*5)

