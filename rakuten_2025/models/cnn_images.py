# System and utilities
import os
import time
import argparse
import json
from dotenv import load_dotenv
# Load environment variables from .env file
# By default, load_dotenv() looks for .env in the current directory or parent directories.
load_dotenv()

# Data manipulation and analysis
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import sys
sys.path.append('../src/functions')
from training_report import get_training_report


## KERAS
from tensorflow.keras.utils import set_random_seed
set_random_seed(66)  # Set random seed for reproducibility
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Garbage collection to free up memory
import gc


def main():
    set_intra_op_parallelism_threads(12)
    set_inter_op_parallelism_threads(4)
    ## input parameters
    parser = argparse.ArgumentParser(description='Train CNN model for product classification using images')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing (square images)')
    parser.add_argument('--conv_filters', nargs='+', type=int, default=[8, 16, 32, 64, 128], help='Number of filters for subsequents Conv2D layers')
    parser.add_argument('--conv_k_reg', type=float, default=0.005, help='Regularization strength for Conv2D layers')
    parser.add_argument('--dense_k_reg', type=float, default=0.02, help='Regularization strength for dense layers')
    args = parser.parse_args()
    
    N_EPOCHS          = args.epochs
    BATCH_SIZE        = args.batch_size
    LR                = args.l_rate
    DATASET_PERC      = args.dataset_perc
    IMG_SIZE          = args.img_size
    CONV_FILTERS_1, CONV_FILTERS_2, CONV_FILTERS_3, CONV_FILTERS_4, CONV_FILTERS_5 = args.conv_filters
    CONV_K_REG        = args.conv_k_reg
    DENSE_K_REG       = args.dense_k_reg
    
    rs = np.random.RandomState(66)
    
    # LOAD DATA:
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_multimodal.csv')
    df = df_full.head(int(df_full.shape[0]*DATASET_PERC))

    hyperparamers_dict = {
        'dataset_percentage': DATASET_PERC,
        'training_rows': int(df_full.shape[0]*DATASET_PERC),
        'epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'img_size': IMG_SIZE,
        'conv_filters': [CONV_FILTERS_1, CONV_FILTERS_2, CONV_FILTERS_3, CONV_FILTERS_4, CONV_FILTERS_5],
        'conv_k_reg': CONV_K_REG,
        'dense_k_reg': DENSE_K_REG
    }

    # Create string labels for generators
    df['prdtypecode_str'] = df['prdtypecode'].astype(str)

    # Split data 
    train_df, val_df = train_test_split(
        df, 
        test_size=0.2, 
        random_state=rs, 
        stratify=df['prdtypecode'])
    
    # Compute class weights
    y_train_for_weights = train_df['prdtypecode'].astype('category').cat.codes
    u_classes = np.unique(y_train_for_weights)
    class_weights = compute_class_weight('balanced', classes=u_classes, y=y_train_for_weights)
    class_weight_dict = {int(cls): round(float(weight), 3) for cls, weight in zip(u_classes, class_weights)}

    # Create data generators that load images from disk
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.15,
        height_shift_range=0.15,
        horizontal_flip=True,
        zoom_range=0.1,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Create generators using string labels
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='imagepath',
        y_col='prdtypecode_str',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=66)
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='imagepath',
        y_col='prdtypecode_str',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=66)
    
    gc.collect()
    
    ### MODEL SETUP:
    input_layer = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # C1
    conv1 = Conv2D(filters=CONV_FILTERS_1, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(input_layer)
    conv1 = BatchNormalization()(conv1)
    conv1 = Conv2D(filters=CONV_FILTERS_1, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv1)
    conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    pool1 = Dropout(0.25)(pool1)
    # C2
    conv2 = Conv2D(filters=CONV_FILTERS_2, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Conv2D(filters=CONV_FILTERS_2, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv2)
    conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    pool2 = Dropout(0.25)(pool2)
    # C3
    conv3 = Conv2D(filters=CONV_FILTERS_3, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Conv2D(filters=CONV_FILTERS_3, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv3)
    conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    pool3 = Dropout(0.25)(pool3)
    # C4
    conv4 = Conv2D(filters=CONV_FILTERS_4, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Conv2D(filters=CONV_FILTERS_4, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv4)
    conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)
    pool4 = Dropout(0.25)(pool4)
    # C5
    conv5 = Conv2D(filters=CONV_FILTERS_5, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Conv2D(filters=CONV_FILTERS_5, kernel_size=(3, 3), 
                   activation='relu', kernel_regularizer=l2(CONV_K_REG))(conv5)
    conv5 = BatchNormalization()(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5)
    pool5 = Dropout(0.25)(pool5)

    # Global Average Pooling
    gap = GlobalAveragePooling2D()(pool5)

    # Dense layers
    # D1
    dense1 = Dense(512, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(gap)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.4)(dense1)
    # D2
    dense2 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)

    # D3
    dense3 = Dense(64, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.3)(dense3)

    # Output layer
    output = Dense(len(df['prdtypecode'].unique()), activation='softmax', kernel_regularizer=l2(0.001))(dense3)

    # Create model
    model = Model(inputs=input_layer, outputs=output)
    f1_metric = F1Score(average='macro', name='f1_score')

    early_stopping = EarlyStopping(monitor='val_f1_score', patience=7, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1, mode='min')

    lr_schedule = CosineDecayRestarts(
    initial_learning_rate=LR,
    first_decay_steps=10,        # First cycle length
    t_mul=1.5,                   # Each cycle gets 1.5x longer
    m_mul=0.6,                   # LR drops 40% after each restart
    alpha=0.000001                 # Minimum LR

    )

    model.compile(optimizer=AdamW(learning_rate=LR, weight_decay=0.01), loss='categorical_crossentropy',
                   metrics=['accuracy', f1_metric])
    print(model.summary())

    print('--'* 50)
    print(f'Using {DATASET_PERC*100}% of the dataset for training ({int(df_full.shape[0]*DATASET_PERC)} rows) .')
    print(f'Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples')
    print(f'Starting Image CNN training with parameters:')
    print(f'------>number of epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, initial learning_rate={LR}') 
    print(f'------>image_size={IMG_SIZE}x{IMG_SIZE}')
    print(f'------>conv_filters=[{args.conv_filters}], conv_k_reg={CONV_K_REG}, dense_k_reg={DENSE_K_REG}')
    print('--'* 50)

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=N_EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=[early_stopping, reduce_lr],
        class_weight=class_weight_dict,
        verbose=1
    )
    print('Image CNN training completed!')
    model_name = f'image-cnn-epochs-{N_EPOCHS}-lr-{LR}_testing_valf1-{history.history["val_f1_score"][-1]:.3f}'
    if history.history['val_f1_score'][-1] > 0.7:
        
        model.save(os.path.join(OUT_PATH, f'{model_name}.keras'))

        print(f'Model saved as {model_name}.keras')

        # Save training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'f1_score': history.history['f1_score'],  # Training F1 score
            'val_f1_score': history.history['val_f1_score'],  # Validation F1 score
            'learning_rate': history.history.get('learning_rate', [LR] * len(history.history['loss']))
        }
        
        with open(os.path.join(OUT_PATH, f'{model_name}_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)

    print("Generating evaluation report...")
    summary, report = get_training_report(
        model=model,
        history=history, 
        validation_data=val_generator,
        save_dir=OUT_PATH,
        model_name=model_name,
        h_parameters=hyperparamers_dict)

    return model, history

if __name__ == '__main__':
    t1 = time.time()
    model, history = main()
    t2 = time.time()
    print(f'Total time taken: {(t2 - t1) / 60:.2f} minutes')