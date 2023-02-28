import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import glob
import matplotlib.pyplot as plt
import cv2
import seaborn as sns
import random
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MultiLabelBinarizer


apple_green_img = glob.glob('./dataset/apple-green/*.png')
apple_red_img = glob.glob('./dataset/apple-red/*.png')
orange_yellow_img = glob.glob('./dataset/orange-yellow/*.png')
banana_yellow_img = glob.glob('./dataset/banana-yellow/*.png')

print('Number of images with fire : {}'.format(len(apple_green_img)))
print('Number of images without fire : {}'.format(len(apple_red_img)))
print('Number of images with fire : {}'.format(len(orange_yellow_img)))
print('Number of images without fire : {}'.format(len(banana_yellow_img)))

lst_images_random = random.sample(apple_green_img,10) + random.sample(apple_red_img,10)+random.sample(orange_yellow_img,10)+random.sample(banana_yellow_img,10)
random.shuffle(lst_images_random)
plt.figure(figsize = (20,20))
for i in range(len(lst_images_random)):
    plt.subplot(4,10,i+1)
    img = cv2.imread(lst_images_random[i])
    img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    plt.imshow(img,cmap = 'gray')
    if "non_fire" in lst_images_random[i]:  
        plt.title('Image')
    else:
        plt.title("Image with")
plt.show()


labels = [
    ("green", "apple"),
    ("red", "apple"),
    ("yellow", "banana"),
    ("orang", "orrange"),
]

mlb = MultiLabelBinarizer()
mlb.fit(labels)
# MultiLabelBinarizer(classes=None, sparse_output=False)
mlb.classes_
print(mlb.classes_)
mlb.transform([("red", "apple")])

print(mlb.transform([("green", "apple")]))
print(mlb.transform([("red", "apple")]))
print(mlb.transform([("yellow", "orrange")]))
print(mlb.transform([("orang", "banana")]))

lst_apple_green = []
for x in apple_green_img:
  lst_apple_green.append([x,[1, 0, 1, 0, 0, 0, 0]])
lst_apple_red = []
for x in apple_red_img:
  lst_apple_red.append([x, [1, 0, 0, 0, 0, 1, 0]])
lst_orrange_yellow = []
for x in orange_yellow_img:
  lst_orrange_yellow.append([x,[0, 0, 0, 0, 1, 0, 1]])
lst_banana_yellow_img = []
for x in banana_yellow_img:
  lst_banana_yellow_img.append([x,[0, 1, 0, 1, 0, 0, 0]])
lst_complete = lst_apple_green + lst_apple_red +lst_orrange_yellow + lst_banana_yellow_img
random.shuffle(lst_complete)

df = pd.DataFrame(lst_complete,columns = ['files','target'])


def preprocessing_image(filepath):
  img = cv2.imread(filepath) 
  img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR) 
  img = cv2.resize(img,(196,196))  
  img = img / 255 
  return img

def create_format_dataset(dataframe):
  X = []
  y = []
  for f,t in dataframe.values:
    X.append(preprocessing_image(f))
    y.append(t)
  
  return np.array(X),np.array(y)

X, y = create_format_dataset(df)


model = Sequential()
model.add(Conv2D(filters=16, kernel_size=(5, 5), activation="relu", input_shape=(196,196,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.1)

model.save('1.h5')

from keras.utils import load_img, img_to_array
img = load_img('dataset/apple-green/rotated_by_15_Screen Shot 2018-06-08 at 4.59.44 PM.png',target_size=(400,400,3))
img = img_to_array(img)
img = img/255



callbacks = [EarlyStopping(monitor = 'val_loss',patience = 5,restore_best_weights=True)]
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs = 30,batch_size = 64,callbacks = callbacks)

model.save('model.h5')
