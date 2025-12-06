# Classifying Galaxies Using Convolutional Neural Networks (CNNs)

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from utils import load_galaxy_data
import app


input_data, labels = load_galaxy_data()
# examine data shape
print(input_data.shape) #prints (1400, 128, 128, 3) 3=RGB
print(labels.shape) #prints (1400, 4) #one-hot vevtors

# split data
x_train, x_test, y_train, y_test = train_test_split(
    input_data, labels,
    test_size=0.2,
    random_state=222,
    stratify=labels,
    shuffle=True
)
# preprocess data
data_generator = ImageDataGenerator(rescale=1./255)
training_iterator = data_generator.flow(x_train, y_train,batch_size=5)
validation_iterator = data_generator.flow(x_test, y_test, batch_size=5)
# build model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128,128,3)))

# add convolutional layers
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation="softmax"))

print(model.summary()) # 7,164 parameters

# compile & optimize model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics=[tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()]
    )

# fit and train model
model.fit(
        training_iterator,
        steps_per_epoch=len(x_train)/5,
        epochs=8,
        validation_data=validation_iterator,
        validation_steps=len(x_test)/5
)
# val_categorical_accuracy: 0.7071 - val_auc: 0.8916
# class correctly identified 70%
# 89% chance of assigning a higher probability to a true class

from visualize import visualize_activations
visualize_activations(model,training_iterator)
'''
Galaxy_0
	Model prediction: [0.19383353 0.18948558 0.27258453 0.34409636]
	True label: Other (3)
	Correct: True
Galaxy_1
	Model prediction: [0.3759237  0.41150066 0.10383832 0.10873733]
	True label: Regular (0)
	Correct: False
Galaxy_2
	Model prediction: [0.35696033 0.20063323 0.1922038  0.2502026 ]
	True label: Merger (2)
	Correct: False
Galaxy_3
	Model prediction: [0.47369608 0.22201233 0.14087941 0.16341217]
	True label: Regular (0)
	Correct: True
Galaxy_4
	Model prediction: [0.08501698 0.01737029 0.7591698  0.13844289]
	True label: Merger (2)
	Correct: True
'''






