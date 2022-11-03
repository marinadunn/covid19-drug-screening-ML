# import modules
# sys
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import numpy as np
import os, sys
import h5py
import warnings
warnings.filterwarnings("ignore")

# Plotting
import matplotlib
import matplotlib.pyplot as plt

# Machine Learning
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, accuracy_score, f1_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Flatten, Conv3D, MaxPool3D
from tensorflow.keras.layers import Dropout, Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
import scipy
from scipy.ndimage import gaussian_filter

print("Scikit-learn Version: ", sklearn.__version__)
print("Scipy Version: ", scipy.__version__)
print("Matplotlib Version: ", matplotlib.__version__)
print("Tensorflow Version: ", tf.__version__)

def get_3D_bounds(xyz):
    """
    Finds min and max values of x, y, and z arrays.
    
    Example:
        >>> import numpy as np
        >>> xyz = np.array([[0, 1], [0, 0.3, 1], [0.1, 0.8, 1]])
        >>> xmin, xmax, ymin, ymax, zmin, zmax = get_3D_bounds(xyz)
        
    Args:
        xyz:
            A numpy ndarray of with 3 columns and row number corresponding to number of atoms
    
    Returns
    -------
        xmin:
            Minimum x coordinate in the array.
        xmax:
            Maximum x coordinate in the array.
        ymin:
            Minimum y coordinate in the array.
        ymax:
            Maximum y coordinate in the array.
        zmin:
            Minimum z coordinate in the array.
        zmax:
            Maximum z coordinate in the array.
    """
    xmin = min(xyz[:, 0])
    xmax = max(xyz[:, 0])
    
    ymin = min(xyz[:, 1])
    ymax = max(xyz[:, 1])
    
    zmin = min(xyz[:, 2])
    zmax = max(xyz[:, 2])
    
    return xmin, xmax, ymin, ymax, zmin, zmax

def voxelizer(xyz, feats):
    """
    Converts x, y, z, coordinate arrays into 3D voxelized data for each ligand.
    
    Args:
        xyz:
            A np.ndarray of columns containing the x, y, and z coordinates for every atom in ligand.
        feats:
            A np.ndarray of columns containing the feature values for every atom in ligand.

    Returns:
        A ndarray of shape (48, 48, 48, 19) representing the input data to be used 
        for model training.
    """
    # Define variables
    vol_dim = [48, 48, 48, 19] 
    atom_radius = 1
    sigma = 0
    
    # Get 3D bounding box for data
    xmin, xmax, ymin, ymax, zmin, zmax = get_3D_bounds(xyz)
    
    # Initialize volume data; create ndarray with dimensions 48x48x48x19
    vol_data = np.zeros((vol_dim[0], vol_dim[1], vol_dim[2], vol_dim[3]), dtype=np.float32)
    
    # Assume same for all axes
    vox_size = float(zmax - zmin) / vol_dim[0]
    
    # Assign each atom to voxels
    num_atoms = xyz.shape[0]
    for i in range(num_atoms):
        x, y, z = xyz[i, 0], xyz[i, 1], xyz[i, 2]
        
        # Make sure coordinate is within volume space
        if (x < xmin or x > xmax) or (y < ymin or y > ymax) or (z < zmin or z > zmax):
            continue
            
        # atom ranges
        cx = (x - xmin) / (xmax - xmin) * (vol_dim[2] - 1)
        cy = (y - ymin) / (ymax - ymin) * (vol_dim[1] - 1)
        cz = (z - zmin) / (zmax - zmin) * (vol_dim[0] - 1)

        vx_from = max(0, int(cx - atom_radius))
        vx_to = min(vol_dim[2] - 1, int(cx + atom_radius))
        
        vy_from = max(0, int(cy - atom_radius))
        vy_to = min(vol_dim[1] - 1, int(cy + atom_radius))
        
        vz_from = max(0, int(cz - atom_radius))
        vz_to = min(vol_dim[0] - 1, int(cz + atom_radius))

        for vz in range(vz_from, vz_to + 1):
            for vy in range(vy_from, vy_to + 1):
                for vx in range(vx_from, vx_to + 1):
                        vol_data[vz, vy, vx, :] += feats[i, :]
    
    # Gaussian filter: 
    if sigma > 0:
        for i in range(vol_data.shape[-1]):
            vol_data[:, :, :, i] = scipy.ndimage.gaussian_filter(vol_data[:, :, :, i], sigma = sigma,
                                                                 truncate = 2 # Truncate filter at this many stdevs
                                                                )
    
    return vol_data

def get_3D_data(input_filepath):
    """
    Reads HDF5 file and returns data and associated labels in the form of np.ndarray.
    
    Example:
        >>> import numpy as np
        >>> import h5py
        >>> dataset = get_3D_data('data.hdf5')
        
    Args:
        filepath:
            A string listing the path where HDF5 file is located with data we 
            want to extract.
    
    Returns
    -------
        data:
            An np.ndarray of extracted data without labels.
        labels:
            An np.ndarray only containing truth labels.
    """
    # Open hdf5 file and loads data
    with h5py.File(input_filepath, 'r') as f:
        
        data = []
        labels = []
        
        # Loop though all the compounds
        for lig_id in f.keys():
            
            # Extract the ligand data, a 100 x 22 np.ndarray; rows correspond to atoms
            ligand_data = f[lig_id]['ligand']
            
            # Remove zero padded rows (ligands with less than 100 atoms)
            num_atoms = 0
            for i in range(ligand_data.shape[0]):
            # if sum of values in row is 0, remove
                if np.sum(np.abs(ligand_data[i, :])) == 0:
                    num_atoms = i
                    break
                    
            # updated ligand data, now num_atoms x 22
            input_data = ligand_data[0:num_atoms, :]
            
            # Ground truth label (0 = no bind, 1 = bind)
            label = f[lig_id].attrs['label']
            
            # First 3 columns represent arrays of x, y, and z coordinates
            xyz = input_data[:, 0:3]
            
            # Remaining 19 columns represent features: 10 one-hot encoded atomic 
            # number columns, Heavy Valence, Hetero Valence, Partial Charge, Mol Code,
            # Hydrophobic, Aromatic, Acceptor, Donor, and Ring
            feats = input_data[:, 3:]
            
            # Create 3D data, shape (48, 48, 48, 19)
            vol_data = voxelizer(xyz, feats)
            
            # Add 3D data + label for each ligand to list
            data.append(vol_data)
            labels.append(label)
            
        # Convert lists to numpy arrays
        data = np.asarray(data, dtype=np.float32)
        labels = np.asarray(labels, dtype=np.float32)
        
        # Close original file
        f.close()
        
    return data, labels

# 3D CNN
def generate_class_weights(train_labels):
    """
    Generates class weights for a dataset given labels and prints them.
    
    Example:
        >>> import numpy as np
        >>> from sklearn.utils.class_weight import compute_class_weight
        >>> X = np.array([1, 2])
        >>> y = np.array([0, 1])
        >>> class_weights = generate_class_weights(y)
        {0: 0.5, 1: 0.5}
        
    Args:
        train_labels:
            A np.ndarray of ground truth labels for training.
            
    Returns:
        A dictionary containing each label and its respective calculated weight.
    """
    class_labels = np.unique(train_labels)
    class_weights = compute_class_weight(class_weight = 'balanced',
                                                      classes = class_labels,
                                                      y = train_labels)
    class_weights = dict(zip(class_labels, class_weights))
    print('Class weights:', class_weights)
    return class_weights

# Defining model layers, build with the Functional API
def create_model(inputs, outputs):

    model = Model(inputs=inputs, outputs=outputs)
    
    return model

# Compile Model
def compile_model(model, learning_rate):
    """
    Compiles the model for training with specified optimizer, loss function, and metrics using the 
    tensorflow.keras.Model().compile() method, and prints a summary of the model 
    architecture.
    
    Args:
        model:
            A tensorflow.keras.models.Model object that includes Keras.Input and Keras.Output objects.
            
    Returns:
        None
    """
    # use Adam optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    loss = 'binary_crossentropy'
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    model.compile(optimizer = optimizer,
                  loss = loss,
                  metrics = metrics
                 )

    model.summary()
    
# Train Model
def train_model(model, 
                train_data, 
                train_labels, 
                batch_size, 
                NUM_EPOCHS, 
                validation_data, 
                validation_labels, 
                class_weights, 
                callbacks
               ):
    """
    Trains the spacified model.
    
    Args:
        model:
            A compiled tensorflow.keras model object ready for training.
        train_data:
            A tensor of data for training.
        train_labels:
            A tensor of labels for training.
        batch_size:
            None or an integer determining the number of samples per batch of computation; defaults
            to '32' if None.
        NUM_EPOCHS:
            An integer of how many epochs to train the model for.
        validation_data:
            A tensor of data for validation.
        validation_labels:
            A tensor of labels for training.
        class_weights:
            A dictionary containing each label and its respective calculated weight.
        callbacks:
            An optional list of keras.callbacks.Callback instances to use throughout model training.
    
    Returns
    -------
        model:
            A trained tensorflow.keras.models.Model object.
        history:
            A Tensorflow History object of a trained model returned by the 
            tensorflow.keras.Model().fit() method of models created from the 
            tensorflow.keras.callbacks.History() callback.
    """
    history = model.fit(X_train, y_train,
                    batch_size = 64,
                    epochs = 100,
                    validation_data = (X_val, y_val),                
                    shuffle = True,
                    class_weight = class_weights,
                    verbose = 1,
                    callbacks = es
            )
    
    return model, history

# Evaluate Model on test data 
def evaluate_model(model, test_data, test_labels):
    """
    Evaluates the trained model on a set of test data and provides a score based on that 
    selected in 'metrics' variable in the function compile_model(), and prints scores.
    
    Args:
        model:
            A trained tensorflow.keras.models.Model object.
        test_data:
            A tensor of previously unseen data to use for evaluating the model.
        test_labels:
            A tensor of previously unseen labels associated with data from 'test_data'.
    
    Returns:
        None
    """
    score = model.evaluate(test_data, test_labels, verbose=True)
    print("%s: %.2f%%" % (model.metrics_names[0], score[0]*100))
    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
    print("%s: %.2f%%" % (model.metrics_names[2], score[2]*100))
    print("%s: %.2f%%" % (model.metrics_names[3], score[3]*100))
    
# Plot training history
def plot_training(model, 
                  history
                 ):
    """
    Create Matplotlib figures plotting the training history for a model configuration.
    
    Args:
        model:
            A trained tensorflow.keras.models.Model object.
        history:
            A Tensorflow History object of a trained model returned by the 
            tensorflow.keras.Model().fit() method of models created from the 
            tensorflow.keras.callbacks.History() callback.
            
    Returns:
        None
    """
    
    # Function to plot the Accuracy
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    # Function to plot the Loss
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = list(range(len(loss)))
    
    # plot accuracy
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, acc, 'navy', label='Accuracy')
    plt.plot(epochs, val_acc, 'deepskyblue', label= "Validation Accuracy")    
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title("Accuracy Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname='~/DSSI/challenge-problem/', format='jpg')
    #plt.show()

    # plot loss
    figsize=(6, 4)
    figure = plt.figure(figsize=figsize)
    plt.plot(epochs, loss, 'red', label='Loss')
    plt.plot(epochs, val_loss, 'lightsalmon', label= "Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title("Loss Training History")
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(fname='~/DSSI/challenge-problem/', format='jpg')
    #plt.show()
    
# View confusion matrix and classification matrix
def make_stats(model,
               test_data, 
               test_labels,
               class_names
              ):
    """
    For a trained model, creates and displays a confusion matrix and classification report.
    
    Args:
        model:
            A trained tensorflow.keras.models.Model object.
        test_data:
            A tensor of previously unseen data to be used to evaluate model.
        test_labels:
            A tensor of previously unseen labels associated with data from 'test_data'.
            
    Returns:
        None
    """
    # Confusion Matrix
    y_pred = np.argmax(model.predict(test_data), axis=1)
    y_test = np.argmax(test_labels, axis=1)
    
    cm = confusion_matrix(y_test, y_pred)
    #cd = sklearn.metrics.ConfusionMatrixDisplay(cm, display_labels = class_names)
    #cd.plot()
    #plt.title('Confusion Matrix')
    #plt.ylabel('True label')
    #plt.xlabel('Predicted label')
    #plt.tight_layout()
    
    # Classification report
    classification_metrics = classification_report(y_test, 
                                                   y_pred, 
                                                   target_names = class_names
                                                  )
    print(classification_metrics + '\n')
    
# Save model
def save_model_data(model):
    """
    Saves model training data and separate file for History object.
    
    Args:
        model:
            A trained tensorflow.keras.models.Model object.
        history:
            A Tensorflow History object of a trained model returned by the 
            tensorflow.keras.Model().fit() method of models created from the 
            tensorflow.keras.callbacks.History() callback.
            
    Returns:
        None
    """
    model.save(filepath = 'model', include_optimizer = True, 
               overwrite = False, save_format = 'h5')

## Main Script
def main():
    # Load hdf5 files and generate 3D data for train, test, and validation datasets
    X_train, y_train = get_3D_data("data/postera_protease2_pos_neg_train.hdf5")
    X_test, y_test = get_3D_data("data/postera_protease2_pos_neg_test.hdf5")
    X_val, y_val = get_3D_data("data/postera_protease2_pos_neg_val.hdf5")

    print("Number of samples in train are:", y_train.shape)
    print("Number of samples in test are:", y_test.shape)
    print("Number of samples in validation are:", y_val.shape)

    # Generate class weights for train dataset
    class_weights = generate_class_weights(y_train)
    
    # Define 3D CNN variables
    class_names = ['No Bind', 'Bind'] # '0': No Bind, '1': Bind
    NUM_CLASSES = len(class_names)
    NUM_FEATURES = 19  # 19 feature dimensions
    NUM_TRAIN = 19533
    NUM_TEST = 1280
    NUM_VALIDATION = 1130
    NUM_TOTAL = NUM_TRAIN + NUM_TEST + NUM_VALIDATION

    # Convert to tensors
    X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    X_test = tf.convert_to_tensor(X_test, dtype=tf.float32)
    y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
    X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

    input_shape = X_train.shape

    # Define callbacks
    es = [EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10),
        ModelCheckpoint(filepath='best_weights.h5', monitor='val_accuracy',
                        mode='max', verbose=0, save_best_only=True, save_freq='epoch')]
    
    ## Model 1 architecture
    inputs = Input(shape=input_shape)
    # Start with 32 5x5x5 filters
    x = Conv3D(filters=32, kernel_size=(5, 5, 5), activation='relu', strides=1, padding='same')(inputs)
    # Try without residual option 1
    x = Conv3D(filters=32, kernel_size=(5, 5, 5), activation='relu', strides=1, padding='same')(x)
    #x = Conv3D(filters=32, kernel_size=(5, 5, 5), activation='relu', strides=1, padding='same')(x)
    # residual option 2
    x = Conv3D(filters=32, kernel_size=(5, 5, 5), activation='relu', strides=1, padding='same')(x)
    #-------------------
    #x = Conv3D(filters=32, kernel_size=(5, 5, 5), activation='relu', strides=1, padding='same')(x)
    #x = MaxPool3D(pool_size=1, strides=1, padding='same')(x)
    # switch to 64 3x3x3 filters
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPool3D(pool_size=1, strides=1, padding='same')(x)
    x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', strides=1, padding='same')(x)
    x = MaxPool3D(pool_size=1, strides=1, padding='same')(x)
    #x = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu', strides=1, padding='same')(x)
    #x = MaxPool3D(pool_size=1, strides=1, padding='same')(x)

    f = Flatten(data_format='channels_last')(x)
    # Use 128 Dense nodes
    d = Dense(128, activation='relu')(f)
    #d = Dense(128, activation='relu')(d)
    # Here is where fusion would normally happen
    outputs = Dense(2, activation='softmax')(d)

    # Create & train
    print("Creating model")
    model1 = create_model(inputs, outputs)
    compile_model(model1, learning_rate = 4.9e-5)
    print("Beginning Model 1 training")
    model1 = train_model(model1, X_train, y_train,
                            batch_size = 64, 
                            NUM_EPOCHS = 150,
                            shuffle = True,
                            validation_data = (X_val, y_val),
                            class_weights = class_weights,
                            callbacks = es
                            )
    
    # Evaluate
    print("Beginning Model 1 testing")
    evaluate_model(model1, X_test, y_test)
    #save_model_data(model1)
    plot_training(model1, history1)
    make_stats(model1, X_test, y_test)
    
if __name__ == "__main__":
    main()
