class Config(object):
    def __init__(self,batch_size=32,
                 save_model_path='model',
                 lr=0.0002,
                 momentum=0.5,
                 g_depths=[1024,512,256,128,3],
                 d_depths=[3,64,128,256,512],
                 is_training=False):
        self.batch_size=batch_size
        self.save_model_path = save_model_path
        self.lr = lr
        self.momentum  = momentum
        self.g_depths = g_depths
        self.d_depths = d_depths
        self.is_training = is_training
