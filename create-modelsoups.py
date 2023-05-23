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
from tensorflow.keras import layers, Sequential
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from itertools import combinations

# Visualization
import seaborn as sns

# Miscellaneous
from tqdm import tqdm
import os
import warnings
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


#### Convert Numpy array datasets to Tensorflow dataset format
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

## 4. Build ViT model and load fine-tuned models
# Function to load and build the ViT-l16 vision transformer model, which is pre-trained on the Imagenet21k dataset.

def create_model(dropout_prob=0):
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
        layers.Dense(128, activation='gelu'),
        layers.Dense(4, activation='softmax')
    ], name='vit_l16')
    
    return vit_l16

# Load the CSV with the paths to all fine-tuned models' weights
model_pool = pd.read_csv("finetune_results.csv", delimiter = ',')

## 5. Create and evaluate Uniform-, Greedy- and Combi Soup.
# Running the model.evaluate() function on our final soups requires the model to be compiled again. Using model.predict() and than manually calculating the desired evaluation metrics is a workaround.

def model_evaluate(model, eval_dataset, eval_labels):
    """
    Returns the accuracy of a Model Soup.
    Args:
    model : Model Soup.
    eval_dataset : Validation- or Test set in TFDS format.
    eval_labels : Labels of corresponding validation- or test set.
    """
    
    # Generate predictions
    preds = model.predict(eval_dataset, verbose = 0)
    preds = preds.argmax(axis = 1)
    
    # Calculate accuracy
    acc = accuracy_score(eval_labels, preds)
    
    return acc


# Additional evaluation function for generating more evaluation insights such as precision, recall, F1-score, and confusion matrix.

def model_evaluate_ext(model, eval_dataset, eval_labels):
    """
    Returns classification report and confusion matrix of a Model Soup.
    Args:
    model : Model Soup.
    eval_dataset : Validation- or Test set in TFDS format.
    eval_labels : Labels of corresponding validation- or test set.
    """
    
    if model == uniform_soup_model:
        cm_title = 'Uniform Soup'
    elif model == greedy_soup_model:
        cm_title = 'Greedy Soup'
    elif model == combi_soup_model:
        cm_title = 'Combi Soup'
    else:
        cm_title = ''
        
    if eval_dataset == valid_dataset:
        cm_data = 'validation set'
    else:
        cm_data = 'test set'
    
    # Generate predictions
    preds = model.predict(eval_dataset, verbose = 0)
    preds = preds.argmax(axis = 1)
    
    # Calculate accuracy
    acc = accuracy_score(eval_labels, preds)
    
    # Generate classification report
    report = classification_report(eval_labels, preds, target_names = labels, digits=4)
    
    # Generate confusion matrix
    cm = confusion_matrix(eval_labels, preds)
    cm = sns.heatmap(cm, annot = True, fmt = ".0f", cmap='Blues', yticklabels = labels, xticklabels = labels)
    cm.set(title=f'Confusion matrix {cm_title} on {cm_data}', xlabel = 'Predicted label', ylabel = 'True label')
    # Save confusion matrix as file
    cm.get_figure().savefig(f'confusion-matrix-{cm_title}-{cm_data}.png', dpi=400, bbox_inches = 'tight')
    
    return preds, acc, report, cm


## 6. Uniform Soup

def uniform_soup(modelpool):
    """
    Returns Uniform soup and accuracy.
    Args:
    modelpool : List, List of individual fine-tuned models (configurations).
    """
    
    soup = []
    
    # Instantiating model
    tf.keras.backend.clear_session()
    vit_l16 = create_model()
    
    # Loop over all models in model pool
    for individual_model in modelpool:
        
        # Load model weights 
        vit_l16.load_weights(individual_model)
        
        # Create list of current model's weights
        model_weights = [np.array(weight) for weight in vit_l16.weights]
        # Add model weights to soup list
        soup.append(model_weights)
        
    # Average all weights 
    uniform_soup = np.array(soup, dtype=object).mean(axis = 0)
    
    # Replace current weights with uniform soup weights
    for w_old, w_uniform_soup in zip(vit_l16.weights, uniform_soup):
        tf.keras.backend.set_value(w_old, w_uniform_soup)
        
    # Evaluate uniform soup on validation set  
    acc = model_evaluate(vit_l16, valid_dataset, valid_labels)
    
    return vit_l16, acc


### Uniform Soup on validation set

# Create Uniform Soup
uniform_soup_model, uniform_val_acc = uniform_soup(model_pool["models"].values)

# Save Uniform Soup
uniform_soup_model.save_weights("models/uniform-soup.h5")

# Evaluate Uniform Soup
uniform_val_preds, _, uniform_val_report, uniform_val_confmatrix = model_evaluate_ext(uniform_soup_model, valid_dataset, valid_labels)

# Save predictions to file
np.save('uniform_val_preds.npy', uniform_val_preds)

print(f'Uniform Soup: Validation accuracy = {uniform_val_acc*100:.6f}%.')
print(f'Best individual model: Validation accuracy = {model_pool["val_acc"][0]*100:.6f}%.')
print(f'Worst individual model: Validation accuracy = {model_pool["val_acc"].iat[-1]*100:.6f}%.')
print('-'*50)
print(uniform_val_report)
print('-'*50)
print(uniform_val_confmatrix)


### Uniform Soup on test set

# Evaluate Uniform Soup on test set  
uniform_test_preds, uniform_test_acc, uniform_test_report, uniform_test_confmatrix = model_evaluate_ext(uniform_soup_model, test_dataset, test_labels)

# Save predictions to file
np.save('uniform_test_preds.npy', uniform_test_preds)

print(f'Uniform Soup: Test accuracy = {uniform_test_acc*100:.6f}%.')
print('-'*50)
print(uniform_test_report)
print('-'*50)
print(uniform_test_confmatrix)


## 7. Greedy Soup

def greedy_soup(modelpool):
    """
    Returns Greedy soup model, ingredients, and validation accuracy.
    Args:
    model_pool : List, List of individual fine-tuned models.
    """
    # Create initial greedy soup with the best individual model (highest val_accuracy)
    greedy_soup =  [modelpool[0]]
    ingredients = [0]
    
    # Load val_accuracy of best individual model
    val_acc = model_pool["val_acc"][0]
    print(f'Best individual model: Validation accuracy = {val_acc*100:.6f}%.')
    
    # Instantiate model
    tf.keras.backend.clear_session()
    vit_l16 = create_model()
    
    # Load weights of best individual model
    vit_l16.load_weights(modelpool[0])
    
    # Loop over the remaining models in model pool
    print(f'Creating potential greedy soups...')
    for index, individual_model in enumerate(modelpool[1:], start=1):
        
        # Create temporary soup 
        temp_soup = greedy_soup.copy()
        temp_soup.append(individual_model)
        
        # Evaluate temporary soup
        model, temp_val_acc = uniform_soup(temp_soup)
        print('Models ' + str(ingredients)[1:-1] + ', ' + str(index) + ': val_accuracy = ' + str(round(temp_val_acc*100,6))+'%')
        
        # Decide whether to include the current model in the soup
        # If val_accuracy of the temporary soup >= val_accuracy soup, add model to soup
        if temp_val_acc > val_acc:
            val_acc = temp_val_acc
            greedy_soup.append(individual_model)
            ingredients.append(index)
            vit_l16 = model
            print(f'Model {index} has been added to the Greedy Soup!')

    return vit_l16, ingredients, val_acc


### Greedy Soup on validation set

# Create Greedy Soup
greedy_soup_model, greedy_ingredients, greedy_val_acc = greedy_soup(model_pool["models"].values)

# Save Greedy Soup
greedy_soup_model.save_weights("models/greedy-soup.h5")

print('-'*50)
print(f'Greedy Soup: Consists of models {greedy_ingredients}.')
print(f'Greedy Soup: Validation accuracy = {greedy_val_acc*100:.6f}%.')

# Evaluate Greedy Soup on validation set  
greedy_valid_preds, _, greedy_valid_report, greedy_valid_confmatrix =  model_evaluate_ext(greedy_soup_model, valid_dataset, valid_labels)

# Save predictions to file
np.save('greedy_val_preds.npy', greedy_valid_preds)

print(greedy_valid_report)
print('-'*50)
print(greedy_valid_confmatrix)


### Greedy Soup on test set

# Evaluate Greedy Soup on test set  
greedy_test_preds, greedy_test_acc, greedy_test_report, greedy_test_confmatrix =  model_evaluate_ext(greedy_soup_model, test_dataset, test_labels)

# Save predictions to file
np.save('greedy_test_preds.npy', greedy_test_preds)

print(f'Greedy Soup: Test accuracy = {greedy_test_acc*100:.6f}%.')
print('-'*50)
print(greedy_test_report)
print('-'*50)
print(greedy_test_confmatrix)


## 8. Combi Soup
# To generate a Combi Soup, every unique combination of fine-tuned models in a model pool will be souped and evaluated. The best soup (combination of models) is called the Combi Soup. The Combi Soup explores more potential solutions (i.e. soups) than a Greedy Soup. Since the Combi Soup tries every unique combination of models, it will always perform as least as good as the Greedy Soup.

def combi_soup(modelpool, n_best):
    """
    Returns Combi soup model, ingredients, and validation accuracy.
    Args:
    model_pool : List, List of individual fine-tuned models.
    n_best: Int, N best individual models to be selected from model pool.
    """
    print(f'Initial model pool = {modelpool}, number of models = {len(modelpool)}.')
    print('-'*50)
    
    # Create model pool of n_best models
    best_pool = modelpool[:n_best]
    print(f'Best models pool = {best_pool}, number of models = {len(best_pool)}.')
    print('-'*50)
    
    # Current best model
    vit = greedy_soup_model
    ingredients = '0'              
    val_acc = model_pool["val_acc"][0]
    
    # Find unique combinations of models (without repetition) in new model pool
    comb_pool = dict()
    for category in range(2,len(best_pool)+1):
        comb_pool[category] = list(set(combinations(best_pool,category)))
    num_comb = sum([len(comb_pool[value]) for value in comb_pool])
    print(f'Combinations best model pool = {comb_pool}, number of models = {num_comb}.')
    
    # Score every combination of models and save results to new dict
    print(f'Creating potential combi soups...')
    comb_results = dict()
    for key, value in comb_pool.items():
        print('-'*50)
        print(f'Evaluating models for category {key}')
        for comb in value:
            # Create list of the weights for each model in the current combination
            weights = []
            for model in comb:
                # Find paths to the weights of each model in current combination
                weights.append(model_pool[model_pool['model_name']==model]['models'].item())
            # Average weights and score model
            soup, current_val_acc = uniform_soup(weights)
            print(f'Models {comb}: Val_accuracy = {current_val_acc*100:.6f}%.')      
            comb_results[comb] = current_val_acc
            
            # Check if current combi soup is best soup
            if current_val_acc > val_acc:
                vit = soup
                val_acc = current_val_acc
                ingredients = comb
    
    print(f'Finished, I have tried all combinations.')
    
    return vit, ingredients, val_acc, comb_results


# Create Combi Soup
combi_soup_model, combi_ingredients, combi_val_acc, comb_results = combi_soup(model_pool["model_name"].values, 8)

# Save Combi Soup
combi_soup_model.save_weights("models/combi-soup.h5")

# Save all combination results
with open('all-combisoup-results_dict', 'wb') as f:
    pickle.dump(comb_results, f, protocol=pickle.HIGHEST_PROTOCOL)
df = pd.DataFrame.from_dict([comb_results])
df.to_csv('all-combisoup-results.csv')

print('-'*50)
print(f'Combi Soup: Consists of models {combi_ingredients}.')
print(f'Combi Soup: Validation accuracy = {combi_val_acc*100:.6f}%.')

# Evaluate Combi Soup on validation set  
combi_valid_preds, _, combi_valid_report, combi_valid_confmatrix =  model_evaluate_ext(combi_soup_model, valid_dataset, valid_labels)

# Save predictions to file
np.save('combi_val_preds.npy', combi_valid_preds)

print(combi_valid_report)
print('-'*50)
print(combi_valid_confmatrix)

### Combi Soup on test set

# Evaluate Combi Soup on test set  
combi_test_preds, combi_test_acc, combi_test_report, combi_test_confmatrix =  model_evaluate_ext(combi_soup_model, test_dataset, test_labels)

# Save predictions to file
np.save('combi_test_preds.npy', combi_test_preds)

print(f'Combi Soup: Test accuracy = {combi_test_acc*100:.6f}%.')
print('-'*50)
print(combi_test_report)
print('-'*50)
print(combi_test_confmatrix)

## 9. Results

# Evaluate best individual model test set (and validation set)
def best_indiv_model():
    best_model = create_model()
    best_model.load_weights(model_pool["models"][0])
    return best_model

best_individual_model = best_indiv_model()

# Compute metrics for best individual model on validation set
bindiv_val_preds, _, bindiv_val_report, bindiv_val_confmatrix = model_evaluate_ext(best_individual_model, valid_dataset, valid_labels)
# Save predictions to file
np.save('bestindiv_val_preds.npy', bindiv_val_preds)
print('Best individual model on val')
print(bindiv_val_report)
print('-'*50)


# Compute metrics for best individual model on validation set
bindiv_test_preds, bindiv_test_acc, bindiv_test_report, bindiv_test_confmatrix = model_evaluate_ext(best_individual_model, test_dataset, test_labels)

# Save predictions to file
np.save('bestindiv_test_preds.npy', bindiv_test_preds)
print('Best individual model on test')
print(bindiv_test_report)
print('-'*50)

print(f'These are all the results!')
print('')
print(f'Best individual model: Validation accuracy = {model_pool["val_acc"][0]*100:.6f}%.')
print(f'Worst individual model: Validation accuracy = {model_pool["val_acc"].iat[-1]*100:.6f}%.')
print(f'Uniform Soup: Validation accuracy = {uniform_val_acc*100:.6f}%.')
print(f'Greedy Soup: Validation accuracy = {greedy_val_acc*100:.6f}%.')
print(f'Combi Soup: Validation accuracy = {combi_val_acc*100:.6f}%.')
print('-'*50)
print(f'Best individual model: Test accuracy = {bindiv_test_acc*100:.6f}%.')
print(f'Uniform Soup: Test accuracy = {uniform_test_acc*100:.6f}%.')
print(f'Greedy Soup: Test accuracy = {greedy_test_acc*100:.6f}%.')
print(f'Combi Soup: Test accuracy = {combi_test_acc*100:.6f}%.')

# Save models with respective validation accuracy to file
model_pool.to_csv("all_results.csv", index = False)
