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

## KERAS - Image Processing  
from tensorflow.keras.applications import MobileNetV2

## KERAS - Model Architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.utils import to_categorical

## KERAS - Setup
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# Garbage collection
import gc

## loading image-only data generator class
from multimodal_generator import ImageDataGeneratorAug

def main():
    set_intra_op_parallelism_threads(8)
    set_inter_op_parallelism_threads(4)
    set_random_seed(66)
    
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description='Train image-only CNN for product classification')
    
    # General training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use')

    parser.add_argument('--dense_size', type=int, default=512, help='Dense layer size (increased for image-only)')
    
    # Image CNN parameters
    parser.add_argument('--img_size', type=int, default=224, help='Image size (224 for MobileNetV2)')
    parser.add_argument('--freeze_base', type=bool, default=True, help='Freeze MobileNetV2 initially')
    parser.add_argument('--use_mobilenet_preprocessing', type=bool, default=True, help='Use MobileNetV2 preprocessing')
    
    # Regularization
    parser.add_argument('--dense_k_reg', type=float, default=0.005, help='Dense layer regularization')
    
    args = parser.parse_args()
    
    # Extract parameters
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.l_rate
    DATASET_PERC = args.dataset_perc
    DENSE_SIZE = args.dense_size
    IMG_SIZE = args.img_size
    FREEZE_BASE = args.freeze_base
    USE_MOBILENET_PREP = args.use_mobilenet_preprocessing
    DENSE_K_REG = args.dense_k_reg
    
    rs = np.random.RandomState(66)
    
    # =============================================================================
    # DATA LOADING - SINGLE DATAFRAME WITH IMAGE PATHS
    # =============================================================================
    
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    
    # Load single dataframe that contains image paths and labels
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_ready.csv')
    
    # Use specified percentage of data
    df = df_full.head(int(df_full.shape[0] * DATASET_PERC))
    
    # Store hyperparameters
    hyperparams_dict = {
        'model_type': 'ImageOnly_MobileNetV2',
        'dataset_percentage': DATASET_PERC,
        'training_rows': len(df),
        'epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'dense_size': DENSE_SIZE,
        'img_size': IMG_SIZE,
        'freeze_base': FREEZE_BASE,
        'use_mobilenet_prep': USE_MOBILENET_PREP
    }
    
    # =============================================================================
    # LABEL PROCESSING
    # =============================================================================
    
    # Convert labels to categorical codes and one-hot encode
    y = df['prdtypecode'].astype('category').cat.codes
    num_classes = len(df['prdtypecode'].unique())
    y_categorical = to_categorical(y, num_classes=num_classes)
    
    # Add categorical labels to dataframe for generator
    df['prdtypecode_categorical'] = list(y_categorical)
    
    # =============================================================================
    # TRAIN/VALIDATION SPLIT
    # =============================================================================
    
    # Split data while maintaining image alignment
    train_idx, val_idx = train_test_split(
        range(len(df)), 
        test_size=0.1, 
        random_state=rs, 
        stratify=y
    )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    # Compute class weights for balanced training
    y_train = y[train_idx]
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = {int(cls): round(float(weight), 3) for cls, weight in zip(unique_classes, class_weights)}
    
    # =============================================================================
    # DATA GENERATORS
    # =============================================================================
    
    # Create training and validation generators
    train_generator = ImageDataGeneratorAug(
        df=train_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_mobilenet_prep=USE_MOBILENET_PREP,
        augment_images=True,
        shuffle=True
    )
    
    val_generator = ImageDataGeneratorAug(
        df=val_df,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_mobilenet_prep=USE_MOBILENET_PREP,
        augment_images=False,
        shuffle=False
    )
    
    gc.collect()
    
    # =============================================================================
    # MODEL ARCHITECTURE - IMAGE BRANCH
    # =============================================================================
    
    # Image input
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # MobileNetV2 backbone (pretrained on ImageNet)
    mobilenet_base = MobileNetV2(
        weights='imagenet',
        include_top=False, 
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        input_tensor=image_input
    )
    
    # Freeze MobileNetV2 layers if specified
    if FREEZE_BASE:
        mobilenet_base.trainable = False
    
    # Extract features from MobileNetV2
    image_features_raw = mobilenet_base.output
    image_features_pooled = GlobalAveragePooling2D(name='image_gap')(image_features_raw)
    
    # =============================================================================
    # MODEL ARCHITECTURE - CLASSIFICATION LAYERS
    # =============================================================================
    
    # Image-specific dense layers for feature processing
    # MobileNetV2 outputs 1280 features after GlobalAveragePooling2D
    image_dense1 = Dense(DENSE_SIZE, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                        name='image_dense1')(image_features_pooled)
    image_dense1 = BatchNormalization(name='image_bn1')(image_dense1)
    image_dense1 = Dropout(0.5, name='image_dropout1')(image_dense1)
    
    # Additional dense layer for better feature extraction
    image_dense2 = Dense(int(DENSE_SIZE/2), activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                        name='image_dense2')(image_dense1)
    image_dense2 = BatchNormalization(name='image_bn2')(image_dense2)
    image_dense2 = Dropout(0.5, name='image_dropout2')(image_dense2)
    
    # Final classification layer
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001),
                       name='classification_output')(image_dense2)
    
    # Create the image-only model
    model = Model(inputs=image_input, outputs=predictions, name='image_only_classifier')
    
    # =============================================================================
    # MODEL COMPILATION AND TRAINING SETUP
    # =============================================================================
    
    # Define metrics and callbacks
    f1_metric = F1Score(average='macro', name='f1_score')
    
    early_stopping = EarlyStopping(
        monitor='val_f1_score', 
        patience=8, 
        restore_best_weights=True, 
        verbose=1, 
        mode='max',
        min_delta=0.01          
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='val_f1_score',  
        factor=0.3,             
        patience=3,              
        min_lr=1e-7, 
        verbose=1,
        min_delta=0.005,
        mode='max'               
    )

    
    # Compile model with optimizer and loss function
    model.compile(
        optimizer=AdamW(learning_rate=LR, weight_decay=0.01),
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_metric]
    )
    
    print(model.summary())
    print(f'Total parameters: {model.count_params():,}')
    print(f'Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}')
    
    print('--' * 50)
    print(f'Using {DATASET_PERC*100}% of dataset ({len(df)} rows)')
    print(f'Training: {len(train_df)} samples, Validation: {len(val_df)} samples')
    print(f'Starting image-only training:')
    print(f'------>epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}')
    print(f'------>image: size={IMG_SIZE}x{IMG_SIZE}, freeze_base={FREEZE_BASE}')
    print(f'------>dense_size={DENSE_SIZE}')
    print('--' * 50)
    
    # =============================================================================
    # MODEL TRAINING
    # =============================================================================
    
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
    
    print('Image-only training completed!')
    
    # =============================================================================
    # MODEL SAVING AND EVALUATION
    # =============================================================================
    max_f1_score = np.max(history.history['val_f1_score']).round(3)
    model_name = f'imageonly-mobilenetv2--lr-{LR}_f1-{max_f1_score}'
    
    if max_f1_score > 0.7:
        model.save(os.path.join(OUT_PATH, f'{model_name}.keras'))
        print(f'Model saved as {model_name}.keras')
        
        # Save training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'f1_score': history.history['f1_score'],
            'val_f1_score': history.history['val_f1_score'],
            'learning_rate': history.history.get('learning_rate', [LR] * len(history.history['loss']))
        }
        
        with open(os.path.join(OUT_PATH, f'{model_name}_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
    
    print('Generating evaluation report...')
    
    # Generate evaluation report for image-only model
    summary, report, predictions = get_training_report(
        model=model,
        history=history,
        validation_data=val_generator,
        save_dir=OUT_PATH,
        model_name=model_name,
        h_parameters=hyperparams_dict
    )
    np.save(os.path.join(OUT_PATH, f'{model_name}_predictions.npy'), predictions)
    
    return model, history


if __name__ == '__main__':
    t1 = time.time()
    model, history = main()
    t2 = time.time()
    print(f'Total time taken: {(t2 - t1) / 60:.2f} minutes')

