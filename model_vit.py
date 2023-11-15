import tensorflow as tf
from vit_keras import vit




#define the model class

class AttractivenessModel_vit(tf.keras.Model):
    def __init__(self):
        super(AttractivenessModel_vit, self).__init__()
        self.pretrained_model = vit.vit_b16(
            image_size=80,
            activation='linear',
            pretrained=True,
            include_top=False,
            pretrained_top=False,
            weights='imagenet21k',

        )

        self.pretrained_model.trainable = False

        # define some convolutional layers , with padding and relu activation
        self.conv1 = tf.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu', input_shape=(80,80,3))
        self.conv2 = tf.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu', input_shape=(80,80,3))
        self.conv3 = tf.keras.layers.Conv2D(3, (3,3), padding='same', activation='relu', input_shape=(80,80,3))


        #define the layers
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, activation='relu')
        self.dense2 = tf.keras.layers.Dense(256, activation='relu')
        self.dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.dense4 = tf.keras.layers.Dense(64, activation='relu')
        self.dense5 = tf.keras.layers.Dense(32, activation='relu')
        self.dense6 = tf.keras.layers.Dense(1, activation='linear')
        
        #define batch normalization
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

    def call(self, inputs): 

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pretrained_model(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.batch_norm1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        output = self.dense6(x)


        
        return output
    
    # def model(self):
    #     x = tf.keras.Input(shape=(80,80,3))
    #     return tf.keras.Model(inputs=[x], outputs=self.call(x))
    
    def summary(self):
        x = tf.keras.Input(shape=(80,80,3))
        return tf.keras.Model(inputs=[x], outputs=self.call(x)).summary()
    
    # def compile(self, optimizer, loss, metrics):
    #     super(AttractivenessModel_vit, self).compile()
    #     self.optimizer = optimizer
    #     self.loss = loss
    #     self.metrics = metrics
        
    #     self.model().compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        
    #     return self.model().compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
    
