# System and utilities
import os
import time
import argparse
import json
import ast
from collections import Counter
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

## KERAS - Text Processing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D
from tensorflow.keras.utils import to_categorical

## KERAS - Image Processing  
from tensorflow.keras.applications import MobileNetV2

## KERAS - Model Architecture
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import F1Score

## KERAS - Setup
from tensorflow.keras.utils import set_random_seed
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

# Garbage collection
import gc

## loading multimodal data generator class to handle both text and image data
from multimodal_generator import MultimodalDataGenerator, MultimodalDataGeneratorAug

def main():
    set_intra_op_parallelism_threads(8)
    set_inter_op_parallelism_threads(4)
    set_random_seed(66)
    
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description='Train multimodal CNN for product classification')
    
    # General training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use')

    parser.add_argument('--dense_size', type=int, default=256, help='Dense layer size for fusion')
    
    # Text CNN parameters
    parser.add_argument('--max_seq_len', type=int, default=60, help='Maximum text sequence length')
    parser.add_argument('--emb_dim', type=int, default=250, help='Text embedding dimension')
    parser.add_argument('--vocab_limit', type=int, default=15000, help='Vocabulary size limit')
    parser.add_argument('--conv_filters', type=int, default=128, help='Text conv filters')
    parser.add_argument('--conv_k_list', nargs='+', type=int, default=[2, 3, 4, 5], help='Text conv kernel sizes')
    
    # Image CNN parameters
    parser.add_argument('--img_size', type=int, default=224, help='Image size (224 for MobileNetV2)')
    parser.add_argument('--freeze_base', type=bool, default=True, help='Freeze MobileNetV2 initially')
    parser.add_argument('--use_mobilenet_preprocessing', type=bool, default=True, help='Use MobileNetV2 preprocessing')
    
    # Regularization
    parser.add_argument('--conv_k_reg', type=float, default=0.005, help='Conv layer regularization')
    parser.add_argument('--dense_k_reg', type=float, default=0.005, help='Dense layer regularization')
    
    args = parser.parse_args()
    
    # Extract parameters
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.l_rate
    DATASET_PERC = args.dataset_perc
    DENSE_SIZE = args.dense_size
    MAX_SEQ_LEN = args.max_seq_len
    EMB_DIM = args.emb_dim
    VOCAB_LIMIT = args.vocab_limit
    CONV_FILTERS = args.conv_filters
    CONV_K_LIST = args.conv_k_list
    IMG_SIZE = args.img_size
    FREEZE_BASE = args.freeze_base
    USE_MOBILENET_PREP = args.use_mobilenet_preprocessing
    CONV_K_REG = args.conv_k_reg
    DENSE_K_REG = args.dense_k_reg
    
    rs = np.random.RandomState(66)
    
    # =============================================================================
    # DATA LOADING - SINGLE DATAFRAME WITH ALL MODALITIES
    # =============================================================================
    
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    
    # Load single dataframe that already contains both text tokens and image paths
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_ready.csv')
    
    # Use specified percentage of data
    df = df_full.head(int(df_full.shape[0] * DATASET_PERC))
    
    # Store hyperparameters
    hyperparams_dict = {
        'model_type': 'Multimodal_TextCNN_MobileNetV2',
        'dataset_percentage': DATASET_PERC,
        'training_rows': len(df),
        'epochs': N_EPOCHS,
        'batch_size': BATCH_SIZE,
        'learning_rate': LR,
        'max_seq_len': MAX_SEQ_LEN,
        'emb_dim': EMB_DIM,
        'vocab_limit': VOCAB_LIMIT,
        'conv_filters': CONV_FILTERS,
        'conv_k_list': CONV_K_LIST,
        'img_size': IMG_SIZE,
        'freeze_base': FREEZE_BASE,
        'use_mobilenet_prep': USE_MOBILENET_PREP
    }
    
    # =============================================================================
    # TEXT PREPROCESSING - TOKENS ALREADY PROCESSED
    # =============================================================================
    
    # Parse already processed token lists from dataframe
    df['comb_tokens_fr'] = df['comb_tokens_fr'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Build vocabulary from existing processed tokens
    all_tokens = []
    for token_list in df['comb_tokens_fr']:
        all_tokens.extend(token_list)
    
    token_counter = Counter(all_tokens)
    most_common_tokens = dict(token_counter.most_common(VOCAB_LIMIT))
    vocab = {word: i+1 for i, word in enumerate(most_common_tokens.keys())}
    vocab_size = len(vocab) + 1  # +1 for padding token (0)
    
    # Convert processed tokens to sequences and pad
    def tokens_to_sequences(token_list):
        return [vocab[token] for token in token_list if token in vocab]
    
    sequences = [tokens_to_sequences(tokens) for tokens in df['comb_tokens_fr']]
    X_text = pad_sequences(sequences, maxlen=MAX_SEQ_LEN, padding='post')
    
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
    
    # Split data while maintaining both text and image alignment
    train_idx, val_idx = train_test_split(
        range(len(df)), 
        test_size=0.1, 
        random_state=rs, 
        stratify=y
    )
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    
    X_text_train = X_text[train_idx]
    X_text_val = X_text[val_idx]
    
    # Compute class weights for balanced training
    y_train = y[train_idx]
    unique_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_train)
    class_weight_dict = {int(cls): round(float(weight), 3) for cls, weight in zip(unique_classes, class_weights)}
    
    # =============================================================================
    # DATA GENERATORS
    # =============================================================================
    
    # Create training and validation generators
    train_generator = MultimodalDataGeneratorAug(
        df=train_df,
        text_sequences=X_text_train,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_mobilenet_prep=USE_MOBILENET_PREP,
        augment_images=True,
        shuffle=True
    )
    
    val_generator = MultimodalDataGeneratorAug(
        df=val_df,
        text_sequences=X_text_val,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_mobilenet_prep=USE_MOBILENET_PREP,
        augment_images=False,
        shuffle=False
    )
    
    gc.collect()
    
    # =============================================================================
    # MODEL ARCHITECTURE - TEXT BRANCH
    # =============================================================================
    
    # Text input and embedding layer
    text_input = Input(shape=(MAX_SEQ_LEN,), name='text_input')
    text_embedding = Embedding(
        input_dim=vocab_size, 
        output_dim=EMB_DIM, 
        trainable=True, 
        embeddings_regularizer=l2(0.001),
        name='text_embedding'
    )(text_input)
    
    # Parallel Conv1D branches with different kernel sizes
    text_conv_branches = []
    for kernel_size in CONV_K_LIST:
        conv = Conv1D(
            filters=CONV_FILTERS, 
            kernel_size=kernel_size, 
            activation='relu', 
            kernel_regularizer=l2(CONV_K_REG),
            name=f'text_conv1d_k{kernel_size}'
        )(text_embedding)
        conv = BatchNormalization(name=f'text_bn_k{kernel_size}')(conv)
        pool = GlobalMaxPooling1D(name=f'text_pool_k{kernel_size}')(conv)
        text_conv_branches.append(pool)
    
    # Concatenate all text conv branches
    text_merged = Concatenate(name='text_concat')(text_conv_branches)
    
    # Text-specific dense layers for feature extraction
    text_dense1 = Dense(128, activation='relu', kernel_regularizer=l2(DENSE_K_REG), 
                       name='text_dense1')(text_merged)
    text_dense1 = BatchNormalization(name='text_bn1')(text_dense1)
    text_features = Dropout(0.4, name='text_dropout1')(text_dense1)
    
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
    
    # Image-specific dense layers for dimensionality reduction
    # MobileNetV2 outputs 1280 features after GlobalAveragePooling2D
    image_dense1 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                        name='image_dense1')(image_features_pooled)
    image_dense1 = BatchNormalization(name='image_bn1')(image_dense1)
    image_features = Dropout(0.5, name='image_dropout1')(image_dense1)
    
    # =============================================================================
    # MODEL ARCHITECTURE - FUSION AND CLASSIFICATION
    # =============================================================================
    
    # Concatenate text and image features
    # text_features: 128 dimensions, image_features: 256 dimensions = 384 total
    fused_features = Concatenate(name='multimodal_concat')([text_features, image_features])
    
    # Shared classification layers operating on fused features
    fusion_dense1 = Dense(DENSE_SIZE, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                         name='fusion_dense1')(fused_features)
    fusion_dense1 = BatchNormalization(name='fusion_bn1')(fusion_dense1)
    fusion_dense1 = Dropout(0.5, name='fusion_dropout1')(fusion_dense1)
    
    fusion_dense2 = Dense(int(DENSE_SIZE/2), activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                         name='fusion_dense2')(fusion_dense1)
    fusion_dense2 = BatchNormalization(name='fusion_bn2')(fusion_dense2)
    fusion_dense2 = Dropout(0.5, name='fusion_dropout2')(fusion_dense2)
    
    # Final classification layer
    predictions = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001),
                       name='classification_output')(fusion_dense2)
    
    # Create the complete multimodal model
    model = Model(inputs=[text_input, image_input], outputs=predictions, name='multimodal_classifier')
    
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
    print(f'Starting multimodal training:')
    print(f'------>epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, lr={LR}')
    print(f'------>text: max_seq_len={MAX_SEQ_LEN}, emb_dim={EMB_DIM}, vocab_size={vocab_size}')
    print(f'------>image: size={IMG_SIZE}x{IMG_SIZE}, freeze_base={FREEZE_BASE}')
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
    
    print('Multimodal training completed!')
    
    # =============================================================================
    # MODEL SAVING AND EVALUATION
    # =============================================================================
    max_f1_score = np.max(history.history['val_f1_score']).round(3)
    model_name = f'multimodal-mobilenetv2--lr-{LR}_f1-{max_f1_score}'
    
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
    
    # Generate evaluation report (may need adaptation for multimodal inputs)
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


'''
Step-by-Step MULTIMODAL DATA GENERATOR Process:
1. Initialization (init)

Gets a DataFrame with product info (text tokens + image paths + labels)
Gets pre-processed text sequences (already converted to numbers)
Sets up batch size, image size, preprocessing options
Creates an index array [0, 1, 2, 3, ...] for all products

2. When Keras asks for a batch (getitem)
A. Pick which products to include:

Takes batch indices (e.g., products 64-127 for batch_size=64)
Creates empty arrays to fill:

text_batch: (64, 60) - 64 products, 60 text tokens each
image_batch: (64, 224, 224, 3) - 64 images, 224x224 RGB
label_batch: (64, 27) - 64 one-hot encoded labels



B. For each product in the batch:
TEXT MODALITY:

Grabs the pre-processed text sequence (already numbers like [15, 847, 23, ...])
Puts it directly into text_batch[i] (no extra processing needed)

IMAGE MODALITY:

Reads the image path from DataFrame
Loads the actual image file from disk
Resizes to 224x224 pixels
Converts to numpy array
If MobileNet preprocessing: Applies preprocess_input() (scales to [-1,1] range with ImageNet normalization)
If standard preprocessing: Divides by 255 (scales to [0,1] range)
If image fails to load: Fills with zeros
Puts processed image into image_batch[i]

LABEL MODALITY:

Gets the one-hot encoded label (like [0,0,1,0,0...] for class 3)
Puts it into label_batch[i]

3. Return the batch:
Returns: (text_batch, image_batch), label_batch

Inputs: Tuple with text + image data
Outputs: Label data
'''