#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
import zipfile
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# In[4]:


##IMPORT DATASETS 
train_dir ="C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\train_set"
test_dir ="C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\test_set"

# Initialize counters
total_train_images = 0
total_test_images = 0

# Count images in train set
for category in os.listdir(train_dir):
    category_dir = os.path.join(train_dir, category)
    num_images = len(os.listdir(category_dir))
    print(f"Train - {category}: {num_images} images")
    total_train_images += num_images

# Count images in test set
for category in os.listdir(test_dir):
    category_dir = os.path.join(test_dir, category)
    num_images = len(os.listdir(category_dir))
    print(f"Test - {category}: {num_images} images")
    total_test_images += num_images

# Calculate total number of images
print(f"Total train images: {total_train_images}")
print(f"Total test images: {total_test_images}")


# In[5]:


##DATA VISUALIZATION FOR TRAIN SET
classes = os.listdir("C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\train_set")
classes = sorted(classes)
print(f"Total classes = {len(classes)}")
print(f"Classes: {classes}")


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Sample dataset
data = {
    'Disease': ['BA-cellulitis', 'BA-impetigo', 'FU-athlete-foot', 'FU-nail-fungus', 
                'FU-ringworm', 'PA-cutaneous-larva-migrans', 'VI-chickenpox', 'VI-shingles']
}

# Create a DataFrame
skin_disease_dataset = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(8, 6))
plt.title('Count of Images', size=16)
sns.countplot(x='Disease', data=skin_disease_dataset)
plt.ylabel('Count', size=12)
plt.xlabel('Diseases', size=12)
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.show()


# In[10]:


##VISUALIZE SOME IMAGES FROM THE TRAINING DATASETS
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing import image
from pathlib import Path 
import random
import cv2

NUM_IMAGES = 3

fig, ax = plt.subplots(nrows = len(classes), ncols = NUM_IMAGES, figsize = (10, 20))

p = 0

for c in classes:
    img_path_class = list(Path(os.path.join(train_dir,c)).glob("*.jpg"))
    img_selected = random.choices(img_path_class, k = NUM_IMAGES)
    for i,j in enumerate(img_selected):
        img_bgr = cv2.imread(str(j))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[p,i].imshow(img_rgb)
        ax[p,i].set_title(f"Class: {c}\nShape: {img_rgb.shape}")
        ax[p,i].axis('off')

    p += 1

fig.tight_layout()
fig.show()


# In[12]:


##DATA VIZUALIZATION FOR TEST SET
classes = os.listdir("C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\test_set")
classes = sorted(classes)
print(f"Total classes = {len(classes)}")
print(f"Classes: {classes}")


# In[13]:


##VIZUALIZE SOME IMAGES FROM THE TEST SET
NUM_IMAGES = 4

fig, ax = plt.subplots(nrows = len(classes), ncols = NUM_IMAGES, figsize = (10, 20))

p = 0

for c in classes:
    img_path_class = list(Path(os.path.join(test_dir,c)).glob("*.jpg"))
    img_selected = random.choices(img_path_class, k = NUM_IMAGES)
    for i,j in enumerate(img_selected):
        img_bgr = cv2.imread(str(j))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        ax[p,i].imshow(img_rgb)
        ax[p,i].set_title(f"Class: {c}\nShape: {img_rgb.shape}")
        ax[p,i].axis('off')

    p += 1

fig.tight_layout()
fig.show()


# In[14]:


##BUILDING THE MODEL ARCHITECTURE
from tensorflow.keras import models, layers
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(8, activation='sigmoid'))

model.summary()


# In[16]:


##DATA PREPROCESSING
import cv2  # You'll need OpenCV or another image processing library
import os

# Define your data directory
data_dir ="C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\data dir"

# List all class subdirectories
class_subdirs = os.listdir(data_dir)

# Initialize empty lists for images and labels
train_images = []
train_labels = []

# Load images and labels
for class_subdir in class_subdirs:
    class_path = os.path.join(data_dir, class_subdir)
    class_images = os.listdir(class_path)
    for image_filename in class_images:
        image_path = os.path.join(class_path, image_filename)
        image = cv2.imread(image_path)  # Read the image using OpenCV (adjust as needed)
        train_images.append(image)
        train_labels.append(class_subdir)

# Convert lists to numpy arrays
train_images = np.array(train_images)
train_labels = np.array(train_labels)

# Normalize pixel values (if not already done)
train_images = train_images.astype('float32') / 255

# Print the shapes of the loaded data (for verification)
print("Train images shape:", train_images.shape)
print("Train labels shape:", train_labels.shape)


# In[18]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_dir = "C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\train_set"

# rescale all images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,              # target dir
    target_size=(150, 150), # resizes all images to 150x150
    #batch_size=20,
    class_mode='categorical'     # Binary labels needed with binary_crossentropy loss
)


# In[19]:


for data_batch, labels_batch in train_generator:
    print(f"Data batch shape: {data_batch.shape}")
    print(f"Labels batch shape: {labels_batch.shape}")
    break 


# In[20]:


batch_images, batch_labels = next(train_generator)


# In[21]:


import tensorflow as tf
integer_labels = tf.argmax(batch_labels, axis=1)


# In[22]:


# Plot the first image in the batch
import matplotlib.pyplot as plt
plt.imshow(batch_images[0])
plt.title(f"Label: {integer_labels[0]}")
plt.show()


# In[24]:


test_dir ="C:\\Users\\sangavi\\OneDrive\\Desktop\\DL final pro\\Skin disease dataset\\skin-disease-datasaet\\test_set"

# rescale all images by 1/255
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,              # target dir
    target_size=(150, 150), # resizes all images to 150x150
    batch_size=20,
    class_mode='categorical')    # Binary labels needed with binary_crossentropy loss


# In[25]:


test_images, test_labels = next(test_generator)


# In[26]:


integer_labels = tf.argmax(test_labels, axis=1)


# In[27]:


##MODEL TRAINING
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'],run_eagerly=True
)


# In[28]:


history = model.fit(
    train_generator,
    epochs=25,
    batch_size=15,
    )


# In[29]:


##MODEL TESTING
test_loss, test_acc = model.evaluate(test_images, test_labels)


# In[30]:


predictions = model.predict(test_images)


# In[31]:


##Classification Report
import numpy as np
from sklearn.metrics import classification_report

# Convert one-hot encoded test labels to class labels
if len(test_labels.shape) > 1:
    test_labels = np.argmax(test_labels, axis=1)

# Convert predictions from one-hot encoded to actual labels
y_pred = np.argmax(predictions, axis=1)

# Generate the classification report
print(classification_report(test_labels, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




