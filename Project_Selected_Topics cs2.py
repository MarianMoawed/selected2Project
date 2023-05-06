#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import utils
from pathlib import Path
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import tensorflow as tf
from tensorflow.keras.utils import *



# In[2]:



# Uploading Dataset
#  D:\level 3\material\myProject\crop_part1
cmap = sns.color_palette("Blues")
# Properties
batch_size = 32
img_height = 224
img_width = 224

train_dir = 'D:\level 3\Data\Training'
test_dir = 'D:\level 3\Data\Testing'



# In[3]:


# Create Training Dataset
train_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[4]:



# Create Validation set
val_ds = tf.keras.utils.image_dataset_from_directory(
  train_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


# In[5]:



# Get labels inferred by the method itself
class_names = train_ds.class_names
print(class_names)


# In[6]:



# Visualize Train Dataset
# Count instances in each class of train dataset
train_ds_labels = []

for _, labels in train_ds:
    for i in range(len(labels)):
        train_ds_labels.append(class_names[labels[i].numpy()])


# In[7]:



# Create a pandas Dataset and apply a few methods
df = pd.DataFrame({'Category': train_ds_labels})
cat = df['Category'].value_counts().index.tolist()
cat = [i.title() for i in cat]
count = df['Category'].value_counts().tolist()


# In[37]:


df


# In[8]:



# Plot distribution of instances in our training data
sns.set(style="whitegrid")
plt.figure(figsize=(6, 8))
plt.pie(count, labels=cat, shadow=True, autopct='%.2f', colors=cmap[::-1])
plt.title("Distribution of Types of Tumour in Train set")
plt.show()


# In[9]:


plt.figure(figsize=(15, 15))
for image_batch, label_batch in train_ds.take(1):
    for i in range(12):
        ax=plt.subplot(3, 4, i+1)
        plt.imshow(image_batch[i].numpy().astype('uint8'))
        plt.title(class_names[label_batch[i]])
        plt.axis('off')


# In[10]:


test_paths = []
test_labels = []

for label in os.listdir(test_dir):
    for image in os.listdir(test_dir+'\\'+label):
        test_paths.append(test_dir+'\\'+label+'/'+image)
        test_labels.append(label)

test_paths, test_labels = shuffle(test_paths, test_labels)


# In[11]:


train_paths = []
train_labels = []

for label in os.listdir(train_dir):
    for image in os.listdir(train_dir+'\\'+label):
        train_paths.append(train_dir+'\\'+label+'/'+image)
        train_labels.append(label)

train_paths, train_labels = shuffle(train_paths, train_labels)



# In[12]:


validation=1142


# In[13]:


plt.figure(figsize=(14,6))
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853']
plt.rcParams.update({'font.size': 14})
plt.pie([len(train_labels)-(validation), len(test_labels), validation],
        labels=['Train','Test','Validation'],
        colors=colors, autopct='%.1f%%', explode=(0,0,0),
        startangle=30);


# In[14]:


normalization_layer = layers.Rescaling(1./255)


# In[15]:


#Building the CNN model
model=keras.Sequential([
    normalization_layer,
    layers.Conv2D(64, 3,input_shape=(224,224,3),padding="same"),
    layers.Activation(tf.nn.relu),
    layers.Conv2D(64, 3),
    layers.BatchNormalization(),
    layers.Conv2D(64, 3,padding="same"),
    layers.Activation(tf.nn.relu),
    layers.MaxPooling2D((2,2)),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Conv2D(64, 3),
    layers.Activation(tf.nn.relu),
    layers.MaxPooling2D((2,2),strides=(2,2)),
    layers.BatchNormalization(),
    layers.Dropout(0.35),
    layers.Conv2D(64, 3),
    layers.Activation(tf.nn.relu),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dropout(0.3),
    layers.Dense(512),
    layers.Activation(tf.nn.relu),
    layers.BatchNormalization(),
    layers.Dense(4, name="outputs")
])


# In[16]:


model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[17]:


model.build(input_shape=(None,224,224,3)) # `input_shape` is the shape of the input data
                         # e.g. input_shape = (None, 32, 32, 3)
model.summary()


# In[20]:


epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    
    
)


# In[26]:


# Read test set
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    seed=123
)


# In[28]:


# Make predictions on test set
predictions = model.predict(test_ds)
# We use tf.nn.softmax as we applied padding in training & task is of multiclass 
# prediction. Threfore, we used `SparseCategoricalCrossentropy` in model.compile()
scores = tf.nn.softmax(predictions[:])


# In[31]:


print(scores)


# In[32]:


for i in range(10):
    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
        .format(class_names[np.argmax(scores[i])], 100 * np.max(scores[i]))
    )


# In[33]:


predictions = model.predict(test_ds)


# In[35]:


test_loss, test_accuracy = model.evaluate(test_ds, batch_size=batch_size)


# In[41]:


print(f"Test Loss:     {test_loss*100}")
print(f"Test Accuracy: {test_accuracy*100}")


# In[ ]:




