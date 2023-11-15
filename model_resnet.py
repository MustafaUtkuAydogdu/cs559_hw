import tensorflow as tf
from vit_keras import vit
from keras.applications import ResNet50



#define the model class

class AttractivenessModel_resnet(tf.keras.Model):
    def __init__(self):
        super(AttractivenessModel_resnet, self).__init__()
        # ResNet-50 backbone (excluding the top classification layer)
        self.resnet_backbone = ResNet50(weights='imagenet', include_top=False, input_shape=(80, 80, 3))

        # Global average pooling to reduce spatial dimensions
        self.global_avg_pooling = tf.keras.layers.GlobalAveragePooling2D()

        # Dense Layers for Regression
        self.dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.batch_norm1 = tf.keras.layers.BatchNormalization()

        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.batch_norm2 = tf.keras.layers.BatchNormalization()

        self.output_layer = tf.keras.layers.Dense(1, activation='linear')  # Linear activation for regression

    def call(self, x):
        x = self.resnet_backbone(x)

        # Global average pooling
        x = self.global_avg_pooling(x)

        # Dense tf.keras.layers for regression
        x = self.dense1(x)
        x = self.batch_norm1(x)

        x = self.dense2(x)
        x = self.batch_norm2(x)

        # Output layer for regression
        output = self.output_layer(x)

        return output



        
        return x
    
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
    
