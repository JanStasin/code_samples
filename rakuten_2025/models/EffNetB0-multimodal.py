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
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array

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

class MultimodalDataGenerator:
    """Custom data generator that handles both text sequences and images simultaneously"""
    
    def __init__(self, df, text_sequences, batch_size, img_size, use_efficientnet_prep=True, 
                 augment_images=False, shuffle=True):
        self.df = df.reset_index(drop=True)
        self.text_sequences = text_sequences
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_efficientnet_prep = use_efficientnet_prep
        self.augment_images = augment_images
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return len(self.df) // self.batch_size
    
    def __getitem__(self, index):
        # Generate batch indices
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Initialize batch arrays
        text_batch = np.zeros((self.batch_size, self.text_sequences.shape[1]))
        image_batch = np.zeros((self.batch_size, self.img_size, self.img_size, 3))
        label_batch = np.zeros((self.batch_size, len(self.df['prdtypecode'].unique())))
        
        # Load each sample in the batch
        for i, idx in enumerate(batch_indices):
            # Load text sequence (already preprocessed)
            text_batch[i] = self.text_sequences[idx]
            
            # Load and preprocess image
            img_path = self.df.iloc[idx]['imagepath']
            try:
                img = load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img)
                
                # Apply preprocessing based on configuration
                if self.use_efficientnet_prep:
                    img_array = preprocess_input(img_array)
                else:
                    img_array = img_array / 255.0
                    
                image_batch[i] = img_array
            except:
                # Handle missing/corrupt images with zeros (will be preprocessed accordingly)
                image_batch[i] = np.zeros((self.img_size, self.img_size, 3))
            
            # Load label (one-hot encoded)
            label = self.df.iloc[idx]['prdtypecode_categorical']
            label_batch[i] = label
        
        return [text_batch, image_batch], label_batch
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle=True"""
        if self.shuffle:
            np.random.shuffle(self.indices)


def main():
    set_intra_op_parallelism_threads(12)
    set_inter_op_parallelism_threads(4)
    set_random_seed(66)
    
    ## Parse command line arguments
    parser = argparse.ArgumentParser(description='Train multimodal CNN for product classification')
    
    # General training parameters
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use')
    
    # Text CNN parameters
    parser.add_argument('--max_seq_len', type=int, default=60, help='Maximum text sequence length')
    parser.add_argument('--emb_dim', type=int, default=128, help='Text embedding dimension')
    parser.add_argument('--vocab_limit', type=int, default=15000, help='Vocabulary size limit')
    parser.add_argument('--conv_filters', type=int, default=64, help='Text conv filters')
    parser.add_argument('--conv_k_list', nargs='+', type=int, default=[2, 3, 4], help='Text conv kernel sizes')
    
    # Image CNN parameters
    parser.add_argument('--img_size', type=int, default=224, help='Image size (224 for EfficientNet)')
    parser.add_argument('--freeze_base', type=bool, default=True, help='Freeze EfficientNet initially')
    parser.add_argument('--use_efficientnet_preprocessing', type=bool, default=True, help='Use EfficientNet preprocessing')
    
    # Regularization
    parser.add_argument('--conv_k_reg', type=float, default=0.01, help='Conv layer regularization')
    parser.add_argument('--dense_k_reg', type=float, default=0.01, help='Dense layer regularization')
    
    args = parser.parse_args()
    
    # Extract parameters
    N_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LR = args.l_rate
    DATASET_PERC = args.dataset_perc
    MAX_SEQ_LEN = args.max_seq_len
    EMB_DIM = args.emb_dim
    VOCAB_LIMIT = args.vocab_limit
    CONV_FILTERS = args.conv_filters
    CONV_K_LIST = args.conv_k_list
    IMG_SIZE = args.img_size
    FREEZE_BASE = args.freeze_base
    USE_EFFICIENTNET_PREP = args.use_efficientnet_preprocessing
    CONV_K_REG = args.conv_k_reg
    DENSE_K_REG = args.dense_k_reg
    
    rs = np.random.RandomState(66)
    
    # =============================================================================
    # DATA LOADING - SINGLE DATAFRAME WITH ALL MODALITIES
    # =============================================================================
    
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    
    # Load single dataframe that already contains both text tokens and image paths
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_multimodal_complete.csv')
    
    # Use specified percentage of data
    df = df_full.head(int(df_full.shape[0] * DATASET_PERC))
    
    # Store hyperparameters
    hyperparams_dict = {
        'model_type': 'Multimodal_TextCNN_EfficientNetB0',
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
        'use_efficientnet_prep': USE_EFFICIENTNET_PREP
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
        test_size=0.2, 
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
    train_generator = MultimodalDataGenerator(
        df=train_df,
        text_sequences=X_text_train,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_efficientnet_prep=USE_EFFICIENTNET_PREP,
        augment_images=True,
        shuffle=True
    )
    
    val_generator = MultimodalDataGenerator(
        df=val_df,
        text_sequences=X_text_val,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        use_efficientnet_prep=USE_EFFICIENTNET_PREP,
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
    text_features = Dropout(0.3, name='text_dropout1')(text_dense1)
    
    # =============================================================================
    # MODEL ARCHITECTURE - IMAGE BRANCH
    # =============================================================================
    
    # Image input
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # EfficientNetB0 backbone (pretrained on ImageNet)
    efficientnet_base = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        input_tensor=image_input
    )
    
    # Freeze EfficientNet layers if specified
    if FREEZE_BASE:
        efficientnet_base.trainable = False
    
    # Extract features from EfficientNet
    image_features_raw = efficientnet_base.output
    image_features_pooled = GlobalAveragePooling2D(name='image_gap')(image_features_raw)
    
    # Image-specific dense layers for dimensionality reduction
    image_dense1 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                        name='image_dense1')(image_features_pooled)
    image_dense1 = BatchNormalization(name='image_bn1')(image_dense1)
    image_features = Dropout(0.3, name='image_dropout1')(image_dense1)
    
    # =============================================================================
    # MODEL ARCHITECTURE - FUSION AND CLASSIFICATION
    # =============================================================================
    
    # Concatenate text and image features
    # text_features: 128 dimensions, image_features: 256 dimensions = 384 total
    fused_features = Concatenate(name='multimodal_concat')([text_features, image_features])
    
    # Shared classification layers operating on fused features
    fusion_dense1 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                         name='fusion_dense1')(fused_features)
    fusion_dense1 = BatchNormalization(name='fusion_bn1')(fusion_dense1)
    fusion_dense1 = Dropout(0.4, name='fusion_dropout1')(fusion_dense1)
    
    fusion_dense2 = Dense(128, activation='relu', kernel_regularizer=l2(DENSE_K_REG),
                         name='fusion_dense2')(fusion_dense1)
    fusion_dense2 = BatchNormalization(name='fusion_bn2')(fusion_dense2)
    fusion_dense2 = Dropout(0.3, name='fusion_dropout2')(fusion_dense2)
    
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
        patience=7, 
        restore_best_weights=True, 
        verbose=1, 
        mode='max'
    )
    
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', 
        factor=0.5, 
        patience=4, 
        min_lr=1e-7, 
        verbose=1, 
        mode='min'
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
    
    model_name = f'multimodal-text-efficientnet-epochs-{N_EPOCHS}-lr-{LR}_valf1-{history.history["val_f1_score"][-1]:.3f}'
    
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
    
    # Generate evaluation report (may need adaptation for multimodal inputs)
    summary, report = get_training_report(
        model=model,
        history=history,
        validation_data=val_generator,
        save_dir=OUT_PATH,
        model_name=model_name,
        h_parameters=hyperparams_dict
    )
    
    return model, history


if __name__ == '__main__':
    t1 = time.time()
    model, history = main()
    t2 = time.time()
    print(f'Total time taken: {(t2 - t1) / 60:.2f} minutes')