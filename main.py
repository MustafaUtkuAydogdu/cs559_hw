import tensorflow as tf
import os
import numpy as np
from model_vit import AttractivenessModel_vit

def prepare_data(directory):
    images = []
    labels  = []

    for file in os.listdir(directory):
        #set label as the first character of the file name
        y = int(file.split('_')[0])
        # image itself
        img = tf.keras.preprocessing.image.load_img(os.path.join(directory,file), target_size=(80,80), color_mode='rgb')
        img = (tf.keras.preprocessing.image.img_to_array(img)) / 255.0
        images.append(img)
        labels.append(y)

    
    return (np.array(images), np.array(labels))


train_path = 'SCUT_FBP5500_downsampled/training'
val_path = 'SCUT_FBP5500_downsampled/validation'
test_path = 'SCUT_FBP5500_downsampled/test'

train_images, train_labels = prepare_data(train_path)
val_images, val_labels = prepare_data(val_path)
test_images, test_labels = prepare_data(test_path)

print(train_images.shape)
print(train_labels.shape)
print(val_images.shape)
print(val_labels.shape)
print(test_images.shape)
print(test_labels.shape)

#conver to tf.data.Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))

#shuffle and batch
BATCH_SIZE = 4



#shuffle and batch
train_dataset = train_dataset.shuffle(1000).batch(batch_size=BATCH_SIZE)
val_dataset = val_dataset.shuffle(1000).batch(batch_size=BATCH_SIZE)
test_dataset = test_dataset.shuffle(1000).batch(batch_size=BATCH_SIZE)


# #plot random images from train_dataset
# import matplotlib.pyplot as plt
# # Extract a batch of images and labels from the train_dataset
# for images, labels in train_dataset.take(1):
#     # Plot a random selection of images (e.g., first 5 images)
#     num_images_to_plot = 5
#     for i in range(num_images_to_plot):
#         plt.subplot(1, num_images_to_plot, i + 1)
#         plt.imshow(images[i].numpy())  # Assuming images are in the correct format
#         plt.title(f"Label: {labels[i].numpy()}")
#         plt.axis('off')

#     plt.show()




#build the model - we are going to try 3 different models 

#model 1 - visual transformer - START
# model_vit = AttractivenessModel_vit()
# learning_rate_vit = 0.001
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_vit)


# model_vit.compile(optimizer='adam', loss='mse', metrics=['mae'])


# print("gpu : ",tf.test.gpu_device_name())
# print(tf.config.list_physical_devices('GPU'))


# #training
# model_vit.fit(train_images,train_labels ,epochs=100, validation_data=val_dataset,verbose=1)


# #evaluate
# model_vit.evaluate(test_images, test_labels, verbose=1)
#model 1 - visual transformer - END


# #do same for resnet
from model_resnet import AttractivenessModel_resnet

model_resnet = AttractivenessModel_resnet()
learning_rate_resnet = 0.001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_resnet)


model_resnet.compile(optimizer='adam', loss='mse', metrics=['mae'])

model_resnet.fit(train_images,train_labels ,epochs=10, validation_data=val_dataset,verbose=1)


#evaluate  
model_resnet.evaluate(test_images, test_labels, verbose=1)

# do same for vanilla cnn
# from model_vanilla_cnn import AttractivenessModel_vanilla_cnn

# model_resnet = AttractivenessModel_vanilla_cnn()
# learning_rate_resnet = 0.001
# optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_resnet)


# model_resnet.compile(optimizer='adam', loss='mse', metrics=['mae'])

# model_resnet.fit(train_images,train_labels ,epochs=100, validation_data=val_dataset,verbose=1)


# #evaluate  
# model_resnet.evaluate(test_images, test_labels, verbose=1)