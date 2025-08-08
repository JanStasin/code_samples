# System and utilities
import os
import time
import argparse
import json
from dotenv import load_dotenv
# # Load environment variables from .env file
# # By default, load_dotenv() looks for .env in the current directory or parent directories.
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
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
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
    parser = argparse.ArgumentParser(description='Train EfficientNetB0 model for product classification using images')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (reduced for EfficientNet)')
    parser.add_argument('--l_rate', type=float, default=0.001, help='Initial learning rate (lower for transfer learning)')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0)')
    parser.add_argument('--img_size', type=int, default=224, help='Image size for resizing (224 for EfficientNetB0)')
    parser.add_argument('--dense_k_reg', type=float, default=0.01, help='Regularization strength for dense layers')
    parser.add_argument('--freeze_base', type=bool, default=True, help='Whether to freeze base model initially')
    parser.add_argument('--effnet_preproc', type=bool, default=True, help='Use EfficientNet preprocessing or standard rescaling')
    args = parser.parse_args()
    
    N_EPOCHS          = args.epochs
    BATCH_SIZE        = args.batch_size
    LR                = args.l_rate
    DATASET_PERC      = args.dataset_perc
    IMG_SIZE          = args.img_size
    DENSE_K_REG       = args.dense_k_reg
    FREEZE_BASE       = args.freeze_base
    USE_EFFNET_PREPROC = args.effnet_preproc
    
    rs = np.random.RandomState(66)
    
    # LOAD DATA:
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_multimodal.csv')
    df = df_full.head(int(df_full.shape[0]*DATASET_PERC))

    hyperparamers_dict = {
        'model_type': 'EfficientNetB0',
        'dataset_percentage': DATASET_PERC,
        'training_rows': int(df_full.shape[0]*DATASET_PERC),
        'epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'img_size': IMG_SIZE,
        'dense_k_reg': DENSE_K_REG,
        'freeze_base': FREEZE_BASE,
        'use_efficientnet_preprocessing': USE_EFFNET_PREPROC
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

    # Create data generators with appropriate preprocessing
    if USE_EFFNET_PREPROC:
        print('Using EfficientNet ImageNet preprocessing')
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest')
        
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        print('Using standard rescaling (0-1 range)')
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
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
    
    ### MODEL SETUP - EfficientNetB0:
    n_classes = len(df['prdtypecode'].unique())
    
    # Load pretrained EfficientNetB0
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    
    # Freeze base model initially if specified
    if FREEZE_BASE:
        base_model.trainable = False
        print('Base model frozen for initial training')
    
    # Add custom classifier on top
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Dense layers similar to original but adapted for transfer learning
    x = Dense(512, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    
    x = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    predictions = Dense(n_classes, activation='softmax', kernel_regularizer=l2(0.001))(x)
    
    # Create model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    f1_metric = F1Score(average='macro', name='f1_score')

    early_stopping = EarlyStopping(monitor='val_f1_score', patience=7, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-7, verbose=1, mode='min')

    lr_schedule = CosineDecayRestarts(
        initial_learning_rate=LR,
        first_decay_steps=10,
        t_mul=1.5,
        m_mul=0.6,
        alpha=0.000001
    )

    model.compile(
        optimizer=AdamW(learning_rate=LR, weight_decay=0.01), 
        loss='categorical_crossentropy',
        metrics=['accuracy', f1_metric]
    )
    
    print(model.summary())
    print(f'Total parameters: {model.count_params():,}')
    print(f'Trainable parameters: {sum([np.prod(v.shape) for v in model.trainable_weights]):,}')

    print('--'* 50)
    print(f'Using {DATASET_PERC*100}% of the dataset for training ({int(df_full.shape[0]*DATASET_PERC)} rows) .')
    print(f'Training set: {len(train_df)} samples, Validation set: {len(val_df)} samples')
    print(f'Starting EfficientNetB0 transfer learning with parameters:')
    print(f'------>number of epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, initial learning_rate={LR}') 
    print(f'------>image_size={IMG_SIZE}x{IMG_SIZE}')
    print(f'------>dense_k_reg={DENSE_K_REG}, freeze_base={FREEZE_BASE}')
    print(f'------>efficientnet_preprocessing={USE_EFFNET_PREPROC}')
    print('--'* 50)

    callbacks = [early_stopping, reduce_lr]

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=N_EPOCHS,
        validation_data=val_generator,
        validation_steps=len(val_generator),
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    
    print('EfficientNetB0 transfer learning completed!')
    model_name = f'efficientnetb0-epochs-{N_EPOCHS}-lr-{LR}_testing_valf1-{history.history["val_f1_score"][-1]:.3f}'
    
    if history.history['val_f1_score'][-1] > 0.7:
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