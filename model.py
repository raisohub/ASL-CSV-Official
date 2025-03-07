'''
The code in model.py was adapted from the following GitHub repository:
https://github.com/mg343/Sign-Language-Detection?tab=readme-ov-file

The dataset associated with training this model was downloaded from Kaggle:
https://www.kaggle.com/datamunge/sign-language-mnist
'''

import pandas as pd
import kagglehub
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout , BatchNormalization
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Download the dataset from Kaggle
path = kagglehub.dataset_download("datamunge/sign-language-mnist")

print("Path to dataset files:", path)

# Reads the csv data from the downloaded files and converts them into a Pandas DataFrame
# This will allow us to train our neural network on the data
train_df = pd.read_csv(path + "/" + "sign_mnist_train.csv")
test_df = pd.read_csv(path + "/" + "sign_mnist_test.csv")

# Separate the labels (or what we want to predict) from the dataset
# This lets us make sure we're not overfitting (we don't want the model to see what we want it to predict)
y_train = train_df['label']
y_test = test_df['label']
del train_df['label']
del test_df['label']

# Modifies the data values:
#   Converts the labels to binary to make it easier for the model to train
#   Converts the features to 28 x 28 pixels
#   Each RGB value is represented on a scale of 0 to 1
label_binarizer = LabelBinarizer()
y_train = label_binarizer.fit_transform(y_train)
y_test = label_binarizer.fit_transform(y_test)

x_train = train_df.values
x_test = test_df.values

x_train = x_train / 255
x_test = x_test / 255

x_train = x_train.reshape(-1,28,28,1)
x_test = x_test.reshape(-1,28,28,1)

datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

midpoint = len(x_test) // 2
x_test, y_test, x_valid, y_valid = x_test[:midpoint], y_test[:midpoint], x_test[midpoint:], y_test[midpoint:]

model = Sequential()

# Convolutional layers
model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
model.add(BatchNormalization())
model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))

# Convolutional -> Linear layers
model.add(Flatten())

# Linear layers
model.add(Dense(units = 512 , activation = 'relu'))
model.add(Dropout(0.3))

# Final layer that converts to different letters
model.add(Dense(units = 24 , activation = 'softmax'))

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
model.summary()

learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1, factor=0.5, min_lr=0.00001)
history = model.fit(datagen.flow(x_train,y_train, batch_size = 128), epochs = 3, validation_data = (x_valid, y_valid), callbacks = [learning_rate_reduction])

# Evaluate the model on the test data. This final accuracy will be your score!
loss, accuracy = model.evaluate(x_test, y_test, verbose=0)

print(f'Loss: {loss:.4f}')
print(f'Accuracy: {accuracy:.4f}')

# Save the model to a file
model.save('model.keras')