# System and utilities
import os
import time
import argparse
import json
# Data manipulation and analysis
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import sys
sys.path.append('../src/functions')
from image_helpers import load_images_batch



## KERAS
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.layers import Dropout, BatchNormalization, Concatenate, Conv2D, GlobalAveragePooling2D
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from keras.metrics import F1Score
from keras.utils import to_categorical

# Image processing


def main():
    ## input parameters
    parser = argparse.ArgumentParser(description='Train CNN model for product classification')
    parser.add_argument('--epochs', type=int, default=15, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--max_seq_len', type=int, default=75, help='Maximum sequence length')
    parser.add_argument('--emb_dim', type=int, default=300, help='Dimension of the embedding layer')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0)')
    parser.add_argument('--vocab_limit', type=int, default=40000, help='Limit on vocabulary size')
    parser.add_argument('--conv_filters', type=int, default=128, help='Number of filters for Conv1D layers')
    parser.add_argument('--conv_k_list', nargs='+', type=int, default=[2, 3, 4, 5], help='List of kernel sizes for Conv1D layers')
    parser.add_argument('--conv_k_reg', type=float, default=0.005, help='Regularization strength for Conv1D layers')
    parser.add_argument('--dense_k_reg', type=float, default=0.02, help='Regularization strength for dense layers')
    parser.add_argument('--img_size', type=int, default=64, help='Image size for resizing (square images)')
    args = parser.parse_args()
    
    ## for all except --conv_k_list, you can pass 
    EMBEDDING_DIM     = args.emb_dim 
    N_EPOCHS          = args.epochs
    BATCH_SIZE        = args.batch_size
    LR                = args.l_rate
    MAX_SEQUENCE_LENGTH = args.max_seq_len
    DATASET_PERC      = args.dataset_perc
    VOCAB_LIMIT       = args.vocab_limit
    CONV_FILTERS      = args.conv_filters
    CONV_K_LIST       = args.conv_k_list ## this is a tricky bit: you need to pass: --conv_k_list 2 3 4 5 (for example) in the command line, without quotes
    CONV_K_REG        = args.conv_k_reg
    DENSE_K_REG       = args.dense_k_reg
    IMG_SIZE          = args.img_size
    
    rs = np.random.RandomState(66)
    
    # LOAD DATA:
    OUT_PATH = '../misc/'
    PROC_DATA_PATH = '../processed_data/'
    df = pd.read_csv(PROC_DATA_PATH + 'X_train.csv')
    df = df.head(int(df.shape[0]*DATASET_PERC))
    
    ## Preprocess text data
    df['comb_tokens'] = df['comb_tokens'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_tokens_filtered = []
    for token_list in df['comb_tokens']:
        all_tokens_filtered.extend(token_list)
    # Count token frequencies
    token_counter = Counter(all_tokens_filtered)
    # Keep only top N most frequent tokens
    most_common_tokens = dict(token_counter.most_common(VOCAB_LIMIT))
    vocab_filtered = {word: i+1 for i, word in enumerate(most_common_tokens.keys())}
    vocab_size_filtered = len(vocab_filtered) + 1

    def tokens_to_sequences_filtered(token_list):
        return [vocab_filtered[token] for token in token_list if token in vocab_filtered]
        
    sequences = [tokens_to_sequences_filtered(tokens) for tokens in df['comb_tokens']]
    X_text = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    # Load and preprocess images
    print('Loading and preprocessing images...')
    X_images = load_images_batch(df['imagepath'].tolist(), target_size=(IMG_SIZE, IMG_SIZE))
    print(f'Loaded {len(X_images)} images with shape {X_images.shape}')

    y = df['prdtypecode'].astype('category').cat.codes
    num_classes = len(df['prdtypecode'].unique())
    
    # Split data (both text and images)
    X_text_train, X_text_val, X_img_train, X_img_val, y_train, y_val = train_test_split(
        X_text, X_images, y, 
        test_size=0.2, 
        random_state=rs, 
        stratify=y)
    
    # Convert to numpy arrays
    X_text_train = np.array(X_text_train)
    X_text_val = np.array(X_text_val)
    X_img_train = np.array(X_img_train)
    X_img_val = np.array(X_img_val)
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Compute class weights
    u_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=u_classes, y=y_train)
    class_weight_dict = {int(cls): round(float(weight), 3) for cls, weight in zip(u_classes, class_weights)}

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    ### MODEL SETUP:
    
    # TEXT BRANCH
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    
    # Embedding layer
    embedding = Embedding(input_dim=vocab_size_filtered, output_dim=EMBEDDING_DIM, 
                          trainable=True, embeddings_regularizer=l2(0.001))(text_input)

    # Parallel Conv1D branches with different kernel sizes
    conv_branches = []
    for kernel_size in CONV_K_LIST:
        # Conv1D branch
        conv = Conv1D(filters=CONV_FILTERS, kernel_size=kernel_size, 
                      activation='relu', kernel_regularizer=l2(CONV_K_REG))(embedding)
        conv = BatchNormalization()(conv)
        pool = GlobalMaxPooling1D()(conv)
        conv_branches.append(pool)

    # Concatenate text branches
    text_features = Concatenate()(conv_branches)
    
    # IMAGE BRANCH
    image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='image_input')
    
    # Image CNN layers
    img_conv1 = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(image_input)
    img_conv1 = BatchNormalization()(img_conv1)
    img_pool1 = MaxPooling1D(pool_size=2)(img_conv1)
    
    img_conv2 = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(img_pool1)
    img_conv2 = BatchNormalization()(img_conv2)
    img_pool2 = MaxPooling1D(pool_size=2)(img_conv2)
    
    img_conv3 = Conv2D(256, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(img_pool2)
    img_conv3 = BatchNormalization()(img_conv3)
    
    # Global average pooling for images
    image_features = GlobalAveragePooling2D()(img_conv3)
    
    # COMBINE TEXT AND IMAGE FEATURES
    combined_features = Concatenate()([text_features, image_features])

    # Dense layers
    # D1
    dense1 = Dense(512, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(combined_features)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    # D2
    dense2 = Dense(256, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.3)(dense2)
    # D3
    dense3 = Dense(128, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense2)
    dense3 = BatchNormalization()(dense3)
    dense3 = Dropout(0.3)(dense3)

    # Output layer
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(dense3)

    # Create model with multiple inputs
    model = Model(inputs=[text_input, image_input], outputs=output)
    f1_metric = F1Score(average='macro', name='f1_score')

    model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy', metrics=['accuracy', f1_metric])
    print(model.summary())
    
    early_stopping = EarlyStopping(monitor='val_f1_score', patience=5, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1, mode='min')

    print('--'* 50)
    print(f'Using {DATASET_PERC*100}% of the dataset for training ({int(df.shape[0]*DATASET_PERC)} rows) .')
    print(f'Starting Multi-modal CNN training with parameters:')
    print(f'------>number of epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, initial learning_rate={LR}') 
    print(f'------>embedding_dim={EMBEDDING_DIM}, max_sequence_lenght={MAX_SEQUENCE_LENGTH}, vocab_limit={VOCAB_LIMIT}')
    print(f'------>conv_filters={CONV_FILTERS}, conv_k_list={CONV_K_LIST}, conv_k_reg={CONV_K_REG}, dense_k_reg={DENSE_K_REG}')
    print(f'------>image_size={IMG_SIZE}x{IMG_SIZE}')
    print('--'* 50)

    history = model.fit(
        [X_text_train, X_img_train], y_train,  # Multiple inputs
        class_weight=class_weight_dict,
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=([X_text_val, X_img_val], y_val),  # Multiple validation inputs
        callbacks=[early_stopping, reduce_lr],
        verbose=1
    )
    
    print('Multi-modal CNN training completed!')

    model_name = f'text-image-cnn-epochs-{N_EPOCHS}-lr-{LR}_testing'
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
        'lr': history.history.get('lr', [LR] * len(history.history['loss']))
    }
    
    with open(os.path.join(OUT_PATH, f'{model_name}_history.json'), 'w') as f:
        json.dump(history_dict, f, indent=2)

    return model, history

if __name__ == "__main__":
    t1 = time.time()
    model, history = main()
    t2 = time.time()
    print(f'Total time taken: {(t2 - t1) / 60:.2f} minutes')