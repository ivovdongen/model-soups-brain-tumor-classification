# View current hardware
#!nvidia-smi

# **Import libraries**

# Data processing
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
import cv2

# Models
from vit_keras import vit
import tensorflow_addons as tfa
from tensorflow.keras import layers, Sequential

# Miscellaneous
from tqdm import tqdm
import os
import warnings
import random
from numpy.random import seed
import gc
import pickle

warnings.filterwarnings('ignore', module='vit_keras')
# Set random seed in Numpy
np.random.seed(24)
# Set random seed in Tensorflow
tf.random.set_seed(24) 

## 1. Load data

data_dir = "/master-thesis/brain-tumor-mri-dataset"

# Create labels
labels = []

for sub_folder in os.listdir(os.path.join(data_dir,'Training')):
    labels.append(sub_folder)
print(f'Labels: {labels}')

# Load and combine data from Training and Testing folder
X_train = []
y_train = []
img_size = 384

def read_data(subset):
  for i in labels:
      folderPath = os.path.join(data_dir,subset,i)
      for j in tqdm(os.listdir(folderPath)):
          img = cv2.imread(os.path.join(folderPath,j))
          img = cv2.resize(img,(img_size, img_size))
          X_train.append(img)
          y_train.append(i)

read_data('Training')
read_data('Testing')
        
X_train = np.array(X_train)
y_train = np.array(y_train)


## 2. Exploratory Data Analysis (EDA)
# The output below shows the shape of the NumPy array containing our dataset. There are 7023 images of size 384x384 pixels with three color channels.
print(f'Data shape {X_train.shape}')


### Number of samples per class
# The output below depicts the class balance. Every one of the four classes accounts for around 1/4 of the total samples, making this dataset reasonably balanced.

# Number of samples per class
np.unique(y_train, return_counts=True)


## 3. Data preprocessing

print(np.min(X_train))
print(np.max(X_train))

X_train = vit.preprocess_inputs(X_train)
print(np.min(X_train))
print(np.max(X_train))


### Create data split
# The size of the training-, validation,- and test set are 76,5%, 13,5%, 10% of the total samples, respectively

# Shuffle data
X_train, y_train = shuffle(X_train, y_train, random_state=24)

# Create initial train (90%) and test (10%) split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=24, stratify=y_train)

# Use 15% of the training set as validation set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, random_state=24, stratify = y_train)


### Label encoding

# Instantiate LabelEncoder function
enc = LabelEncoder()

y_train = enc.fit_transform(y_train)
y_val = enc.transform(y_val)
y_test = enc.transform(y_test)

# Convert class vector (integers) to binary class matrix
y_train = to_categorical(y_train, num_classes = 4)
y_val = to_categorical(y_val, num_classes = 4)
y_test = to_categorical(y_test, num_classes = 4)


### Convert Numpy array datasets to Tensorflow dataset format
# [Tensorflow datasets](http://https://www.tensorflow.org/api_docs/python/tf/data/Dataset) allow for easy batching, prefetching, and mapping data augmentation.

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
valid_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))


### Batch and prefetch the images
# The '[batch](http://https://www.tensorflow.org/api_docs/python/tf/data/Dataset#batch)' and '[prefetch](http://https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)' functions from Tensorflow deliver data for the next training step before the current step has finished, creating an [efficient input pipeline](http://https://www.tensorflow.org/guide/data_performance).

batch_size = 16

train_dataset = train_dataset.batch(batch_size, drop_remainder= True)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
valid_dataset = valid_dataset.batch(batch_size, drop_remainder= True)
valid_dataset = valid_dataset.prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size, drop_remainder= False)
test_dataset = test_dataset.prefetch(tf.data.AUTOTUNE)


### Extract labels
# Extract the labels of the validation- and test set in order to evaluate our Model Soups later.

valid_labels = np.concatenate([y for x, y in valid_dataset], axis=0)
test_labels = np.concatenate([y for x, y in test_dataset], axis=0)

# Convert one-hot labels back to single-digit labels
valid_labels=np.argmax(valid_labels, axis=1)
test_labels=np.argmax(test_labels, axis=1)

np.save('validation_labels.npy', valid_labels)
np.save('test_labels.npy', test_labels)

## 4. Load and build ViT model
# Function to load and build the ViT-l16 vision transformer model, which is pre-trained on the Imagenet21k dataset.


def create_model(dropout_prob=0, w_init='glorot_normal', b_init='zeros'):
    """
    Returns pre-trained ViT-l16 model  
    Args:
    dropout_prob : Float, Dropout probability.
    w_init: Matrix of weights to be initialized in the first dense layer.
    b_init: Matrix of bias' to be initialized in in the first dense layer.
    """

    ## Load ViT-l16 model
    feature_extractor = vit.vit_l16(
        image_size = img_size,
        activation = 'softmax',
        pretrained = True,
        include_top = False,
        pretrained_top = False,
        classes = 4)
    
    for layer in feature_extractor.layers:
      layer.trainable = False
    
    vit_l16 = Sequential([
        layers.Input(shape=(384,384,3), name='input_image'),
        feature_extractor,
        layers.Dropout(dropout_prob),
        layers.Dense(128, activation='gelu', kernel_initializer=w_init, bias_initializer=b_init),
        layers.Dense(4, activation='softmax')
    ], name='vit_l16')
    
    return vit_l16


## 5. Create hyperparameter grid for Model Soups

# The [original Model Soups paper by Wortsman et al.](http://https://arxiv.org/abs/2203.05482) fine-tuned on six hyperparameters.
# 
# - Learning rate;
# - Weight decay;
# - Iterations (epochs);
# - Data augmentation intensity;
# - Mixup;
# - and Label smoothing.


# Define functions for three different intensities of data augmentation
def img_aug_low(image, label):
    """
    Tensorflow DS map function to augument images randomly
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    return image, label 

def img_aug_medium(image, label):
    """
    Tensorflow DS map function to augument images randomly
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_flip_up_down(image)
    return image, label 

def img_aug_high(image, label):
    """
    Tensorflow DS map function to augument images randomly
    """
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, 0.15)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    return image, label   


### Create hyperparameter grid

initialization = pd.DataFrame({
    "epochs": 20,
    "learning_rate": [5e-4],
    "weight_decay" : [1e-8],
    "img_aug" : [img_aug_medium],
    "label_smoothing" : [0.1],
    "dropout_prob" : [0.1]
})

num_models = 25

learning_rate = [1e-3, 3e-3, 1e-4, 3e-4, 5e-4, 7e-4]
weight_decay = [1e-5, 1e-6, 1e-7, 1e-8]
epochs = [14, 16, 18, 20]
img_aug = [None, img_aug_low, img_aug_medium, img_aug_high]
label_smoothing = [0, 0.01, 0.1, 0.15, 0.2]
dropout_prob = [0.05, 0.1, 0.15, 0.2, 0.25]

# Create dictonary of random search on hyperparameters above
parameters = [ {
    "epochs": random.choice(epochs),
    "learning_rate": random.choice(learning_rate),
    "weight_decay" : random.choice(weight_decay),
    "img_aug" : random.choice(img_aug),
    "label_smoothing" : random.choice(label_smoothing),
    "dropout_prob" : random.choice(dropout_prob)
} for count in range(num_models)]

model_pool = pd.DataFrame(parameters)
model_pool = pd.concat([initialization, model_pool])
model_pool.to_csv("configurations.csv", index = False)


## 6. Fine-tuning models

### Defining callbacks
# [EarlyStopping](http://https://keras.io/api/callbacks/early_stopping/) monitors the given loss function at end of every epoch. If the loss does not decrease for a 'patience' number of epochs, training terminates.
# [ReduceLROnPlateau](http://https://keras.io/api/callbacks/reduce_lr_on_plateau/) monitors a given metric (e.g. loss). If no improvement is seen for a 'patience' number of epochs, the learning rate is reduced by a given factor.

# Define Early Stopping callback
#early_stopping_callback = tf.keras.callbacks.EarlyStopping(
#    monitor='val_accuracy', 
#    patience=4, 
#    restore_best_weights=True,
#    verbose=1)

# Define Reduce Learning Rate callback
reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',
    patience=2,
    factor=0.1,
    verbose=1)

callbacks_list = [reduce_lr_callback]


### Model training
# Function to train initializer
def train_initializer(train_dataset,
                valid_dataset,
                learning_rate, 
                weight_decay,
                epochs,
                img_aug,
                label_smoothing,
                dropout_prob,
                save_dir = "models/"):
    """
    Returns saved trained model's path, validation evaluation scores and weights and biases
    Args:
    train_ds : Obj, Training set.
    test_ds : Obj, Validation set.
    learning_rate : Float, Learning rate.
    weight_decay : Float, AdamW weight decay.
    epochs : Int, Number of epochs.
    img_aug: Function, Mapping chosen data augmentation intensity.
    label_smoothing: Float, label smoothing for loss function.
    dropout_prob: Float, dropout probability
    save_dir : Str, Model save directory
    """
    # Create directory for saving models
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Apply chosen data augmentation intensity
    train_dataset = train_dataset.unbatch().map(img_aug).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    tf.keras.backend.clear_session()
    vit_l16 = create_model(dropout_prob)
    
    # Compile model
    vit_l16.compile(
            optimizer = tfa.optimizers.AdamW(learning_rate = learning_rate, weight_decay = weight_decay),
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing),
            metrics = ["accuracy"]
        )
    
    
    # Train model
    vit_l16.fit(
        train_dataset,
        validation_data = valid_dataset,
        epochs = epochs,
        callbacks = callbacks_list
    )
    
    # Save weights and biases to use as shared initialization in future fine-tuning
    saved_w = vit_l16.layers[2].get_weights()[0]
    saved_w_init = tf.constant_initializer(saved_w)

    saved_b = vit_l16.layers[2].get_weights()[1]
    saved_b_init = tf.constant_initializer(saved_b)
    
    # Evaluate model
    val_loss, val_acc = vit_l16.evaluate(valid_dataset)
    
    # Save model
    if img_aug == None:
        aug = "None"
    else:
        aug = img_aug.__name__

    model_save_path = save_dir + "initializer-lr" + str(learning_rate) + "_wd" + str(weight_decay) + "_ep" + str(epochs) + "_" + str(aug) + "_ls" + str(label_smoothing) + "_do" + str(dropout_prob) + ".h5"
    if not os.path.isdir(save_dir):
        vit_l16.save_weights(model_save_path)
    else:
        # If model with same parameters already exists
        model_save_path = save_dir + "initializer-lr" + str(learning_rate) + "_wd" + str(weight_decay) + "_ep" + str(epochs) + "_" + str(aug) + "_ls" + str(label_smoothing) + "_do" + str(dropout_prob) + "_" + str(random.choice(np.arange(0,10))) + ".h5"
        vit_l16.save_weights(model_save_path)
    
    
    # Clear GPU memory
    del vit_l16
    gc.collect()
    return model_save_path, val_acc, val_loss, saved_w_init, saved_b_init

# Train initializer
model_paths = []
valid_scores = []
valid_losses = []

model_save_path, val_acc, val_loss, w_initialization, b_initialization = train_initializer(train_dataset,
                                              valid_dataset,
                                              initialization["learning_rate"][0],
                                              initialization["weight_decay"][0],
                                              initialization["epochs"][0],
                                              initialization["img_aug"][0],
                                              initialization["label_smoothing"][0],
                                              initialization["dropout_prob"][0],
                                              save_dir = "models/")

model_paths.append(model_save_path)
valid_scores.append(val_acc)
valid_losses.append(val_loss)


# Function to train each hyperparameter configuration defined above.

def train_model(train_dataset,
                valid_dataset,
                learning_rate, 
                weight_decay,
                epochs,
                img_aug,
                label_smoothing,
                dropout_prob,
                save_dir = "models/"):
    """
    Returns saved trained model's path and validation evaluation scores
    Args:
    train_ds : Obj, Training set.
    test_ds : Obj, Validation set.
    learning_rate : Float, Learning rate.
    weight_decay : Float, AdamW weight decay.
    epochs : Int, Number of epochs.
    img_aug: Function, Mapping chosen data augmentation intensity.
    label_smoothing: Float, label smoothing for loss function.
    dropout_prob: Float, dropout probability
    save_dir : Str, Model save directory
    """
    # Create directory for saving models
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Apply chosen data augmentation intensity
    if img_aug is not None:
        train_dataset = train_dataset.unbatch().map(img_aug).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # Build model
    tf.keras.backend.clear_session()
    vit_l16 = create_model(dropout_prob, w_init=w_initialization, b_init=b_initialization)
    
    # Compile model
    vit_l16.compile(
            optimizer = tfa.optimizers.AdamW(learning_rate = learning_rate, weight_decay = weight_decay),
            loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing = label_smoothing),
            metrics = ["accuracy"]
        )
    
    
    # Train model
    history = vit_l16.fit(
        train_dataset,
        validation_data = valid_dataset,
        epochs = epochs,
        callbacks = callbacks_list
    )
    
    # Evaluate model
    val_loss, val_acc = vit_l16.evaluate(valid_dataset)
    
    # Save model weights
    if img_aug == None:
        aug = "None"
    else:
        aug = img_aug.__name__

    model_save_path = save_dir + "model-lr" + str(learning_rate) + "_wd" + str(weight_decay) + "_ep" + str(epochs) + "_" + str(aug) + "_ls" + str(label_smoothing) + "_do" + str(dropout_prob) + ".h5"
    if not os.path.isdir(save_dir):
        vit_l16.save_weights(model_save_path)
    else:
        # If model with same parameters already exists
        model_save_path = save_dir + "model-lr" + str(learning_rate) + "_wd" + str(weight_decay) + "_ep" + str(epochs) + "_" + str(aug) + "_ls" + str(label_smoothing) + "_do" + str(dropout_prob) + "_" + str(random.choice(np.arange(0,10))) + ".h5"
        vit_l16.save_weights(model_save_path)
        
    # Clear GPU memory
    del vit_l16
    gc.collect()
    return model_save_path, val_acc, val_loss, history

best_valacc = 0
best_history = ''

# Train each hyperparameter configuration defined above.
for config in tqdm(parameters):
    model_save_path, valid_acc, valid_loss, curr_history = train_model(train_dataset, valid_dataset,
                                             config["learning_rate"],
                                             config["weight_decay"],
                                             config["epochs"],
                                             config["img_aug"],
                                             config["label_smoothing"],
                                             config["dropout_prob"],
                                             save_dir = "models/")
    model_paths.append(model_save_path)
    valid_scores.append(valid_acc)
    valid_losses.append(valid_loss)

    if valid_acc > best_valacc:
        best_valacc = valid_acc
        best_history = curr_history

with open('history_best-individual-model.pkl', 'wb') as fp:
    pickle.dump(best_history.history, fp)


### Save trained models
# Create a dataframe containing the configurations, paths, and validation accuracy for each trained model. Export the dataframe to a CSV file.

# Save model paths and scores to df    
model_pool["models"] = model_paths
model_pool["val_acc"] = valid_scores
model_pool["val_loss"] = valid_losses

# Sorting models based on validation accuracy in descending order (best model on top)
model_pool.sort_values(by = "val_acc", ascending = False, inplace = True)
model_pool.reset_index(drop = True, inplace = True)
model_pool['model_name'] =  model_pool.index
first_column = model_pool.pop('model_name')
model_pool.insert(0, 'model_name', first_column)

# Save models (weights) with respective validation accuracy to file
model_pool.to_csv("finetune_results.csv", index = False)
model_pool.to_pickle("finetune_df.pkl")
