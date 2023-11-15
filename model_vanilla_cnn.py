import tensorflow as tf
from vit_keras import vit



#define the model class

class AttractivenessModel_vanilla_cnn(tf.keras.Model):
    def __init__(self):
        super(AttractivenessModel_vanilla_cnn, self).__init__()

        # Convolutional Block 1
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=(64, 64, 3))
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.activation1 = tf.keras.layers.Activation('relu')
        self.pool1 = tf.keras.layers.MaxPooling2D((2, 2))

        # Convolutional Block 2
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), padding='same')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.activation2 = tf.keras.layers.Activation('relu')
        self.pool2 = tf.keras.layers.MaxPooling2D((2, 2))

        # Convolutional Block 3
        self.conv3 = tf.keras.layers.Conv2D(128, (3, 3), padding='same')
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.activation3 = tf.keras.layers.Activation('relu')
        self.pool3 = tf.keras.layers.MaxPooling2D((2, 2))

        # Flatten the output for dense layers
        self.flatten = tf.keras.layers.Flatten()

        # Dense Layers for Regression
        self.dense1 = tf.keras.layers.Dense(128)
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.activation4 = tf.keras.layers.Activation('relu')

        self.dense2 = tf.keras.layers.Dense(64)
        self.batch_norm5 = tf.keras.layers.BatchNormalization()
        self.activation5 = tf.keras.layers.Activation('relu')

        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = self.activation1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.activation2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.activation3(x)
        x = self.pool3(x)

        x = self.flatten(x)

        x = self.dense1(x)
        x = self.batch_norm4(x)
        x = self.activation4(x)

        x = self.dense2(x)
        x = self.batch_norm5(x)
        x = self.activation5(x)

        # Output layer for regression
        output = self.output_layer(x)

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
    
