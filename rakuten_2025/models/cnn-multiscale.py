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
import ast
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from collections import Counter
import sys
sys.path.append('../src/functions')
from training_report import get_training_report
from importance_analyzer import importance_analysis

## KERAS
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.layers import Dropout, BatchNormalization, Concatenate
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.metrics import F1Score
from tensorflow.keras.utils import to_categorical
from tensorflow.config.threading import set_intra_op_parallelism_threads, set_inter_op_parallelism_threads

## Garbage collection
import gc


def main():
    set_intra_op_parallelism_threads(8)
    set_inter_op_parallelism_threads(4)
    ## input parameters
    parser = argparse.ArgumentParser(description='Train CNN model for product classification')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--l_rate', type=float, default=0.005, help='Initial learning rate')
    parser.add_argument('--dense_size', type=int, default=256, help='Dense layer size for fusion')
    parser.add_argument('--max_seq_len', type=int, default=60, help='Maximum sequence length')
    parser.add_argument('--emb_dim', type=int, default=180, help='Dimension of the embedding layer')
    parser.add_argument('--dataset_perc', type=float, default=1.0, help='Percentage of dataset to use for training (0.0 to 1.0)')
    parser.add_argument('--vocab_limit', type=int, default=15000, help='Limit on vocabulary size')
    parser.add_argument('--conv_filters', type=int, default=128, help='Number of filters for Conv1D layers')
    parser.add_argument('--conv_k_list', nargs='+', type=int, default=[2, 3, 4, 5], help='List of kernel sizes for Conv1D layers')
    parser.add_argument('--conv_k_reg', type=float, default=0.005, help='Regularization strength for Conv1D layers')
    parser.add_argument('--dense_k_reg', type=float, default=0.8, help='Regularization strength for dense layers')
    args = parser.parse_args()
    
    ## for all except --conv_k_list, you can pass 
    EMBEDDING_DIM     = args.emb_dim 
    N_EPOCHS          = args.epochs
    BATCH_SIZE        = args.batch_size
    LR                = args.l_rate
    MAX_SEQUENCE_LENGTH = args.max_seq_len
    DENSE_SIZE = args.dense_size
    DATASET_PERC      = args.dataset_perc
    VOCAB_LIMIT       = args.vocab_limit
    CONV_FILTERS      = args.conv_filters
    CONV_K_LIST       = args.conv_k_list ## this is a tricky bit: you need to pass: --conv_k_list 2 3 4 5 (for example) in the command line, without quotes
    CONV_K_REG        = args.conv_k_reg
    DENSE_K_REG       = args.dense_k_reg

    
    rs = np.random.RandomState(66)
    
    # LOAD DATA:
    OUT_PATH = '../misc/reports'
    PROC_DATA_PATH = '../processed_data/'
    df_full = pd.read_csv(PROC_DATA_PATH + 'X_train_with_labels_ext.csv') #'X_train_ready.csv')
    df = df_full.head(int(df_full.shape[0]*DATASET_PERC))
    ## Preprocess the 'comb_tokens' column

    hyperparams_dict = {
    'dataset_percentage': DATASET_PERC,
    'training_rows': int(df_full.shape[0]*DATASET_PERC),
    'n_epochs': N_EPOCHS,
    'batch_size': BATCH_SIZE,
    'initial_learning_rate': LR,
    'dense_size': DENSE_SIZE,
    'embedding_dim': EMBEDDING_DIM,
    'max_sequence_length': MAX_SEQUENCE_LENGTH,
    'vocab_limit': VOCAB_LIMIT,
    'conv_filters': CONV_FILTERS,
    'conv_k_list': CONV_K_LIST,
    'conv_k_reg': CONV_K_REG,
    'dense_k_reg': DENSE_K_REG}

    df['comb_tokens_fr'] = df['comb_tokens_fr'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    all_tokens_filtered = []
    for token_list in df['comb_tokens_fr']:
        all_tokens_filtered.extend(token_list)
    # Count token frequencies
    token_counter = Counter(all_tokens_filtered)
    # Keep only top N most frequent tokens
    most_common_tokens = dict(token_counter.most_common(VOCAB_LIMIT))
    vocab_filtered = {word: i+1 for i, word in enumerate(most_common_tokens.keys())}
    vocab_size_filtered = len(vocab_filtered) + 1

    def tokens_to_sequences_filtered(token_list):
        return [vocab_filtered[token] for token in token_list if token in vocab_filtered]
        
    sequences = [tokens_to_sequences_filtered(tokens) for tokens in df['comb_tokens_fr']]
    X = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')


    y = df['prdtypecode'].astype('category').cat.codes
    num_classes = len(df['prdtypecode'].unique())
    
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.1, 
        random_state=rs, 
        stratify=y)
    
    # This step is necessary for class weights to work correctly:
    X_train = np.array(X_train)
    X_val = np.array(X_val) 
    y_train = np.array(y_train)
    y_val = np.array(y_val)

    # Compute class weights
    u_classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=u_classes, y=y_train)
    class_weight_dict = {int(cls): round(float(weight), 3) for cls, weight in zip(u_classes, class_weights)}
    #print(f'Class weights: {class_weight_dict}')

    y_train = to_categorical(y_train, num_classes=num_classes)
    y_val = to_categorical(y_val, num_classes=num_classes)

    ### MODEL SETUP:
    input_layer = Input(shape=(MAX_SEQUENCE_LENGTH,))

    # Embedding layer
    embedding = Embedding(input_dim=vocab_size_filtered, output_dim=EMBEDDING_DIM, 
                          trainable=True, embeddings_regularizer=l2(0.001))(input_layer)

    # Parallel Conv1D branches with different kernel sizes
    conv_branches = []
    for kernel_size in CONV_K_LIST:
        # Conv1D branch
        conv = Conv1D(filters=CONV_FILTERS, kernel_size=kernel_size, 
                      activation='relu', kernel_regularizer=l2(CONV_K_REG))(embedding)
        conv = BatchNormalization()(conv)
        pool = GlobalMaxPooling1D()(conv)
        conv_branches.append(pool)

    # Concatenate branches
    merged = Concatenate()(conv_branches)

    # Dense layers
    # D1
    dense1 = Dense(DENSE_SIZE, activation='relu', kernel_regularizer=l2(DENSE_K_REG))(merged)
    dense1 = BatchNormalization()(dense1)
    dense1 = Dropout(0.5)(dense1)
    # D2
    dense2 = Dense(int(DENSE_SIZE/2), activation='relu', kernel_regularizer=l2(DENSE_K_REG))(dense1)
    dense2 = BatchNormalization()(dense2)
    dense2 = Dropout(0.5)(dense2)

    # dense3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dense2)
    # dense3 = BatchNormalization()(dense3)
    # dense3 = Dropout(0.3)(dense3)

    # Output layer
    output = Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.001))(dense2)

    # Create model
    model = Model(inputs=input_layer, outputs=output)
    gc.collect()
    f1_metric = F1Score(average='macro', name='f1_score')

    early_stopping = EarlyStopping(monitor='val_f1_score', patience=7, restore_best_weights=True, verbose=1, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_f1_score', factor=0.6, patience=3, min_lr=1e-7, verbose=1, mode='max', min_delta=0.005)

    lr_schedule = CosineDecayRestarts(
    initial_learning_rate=LR,
    first_decay_steps=10,        
    t_mul=1.5,                   
    m_mul=0.6,                   
    alpha=0.000001                 
    )

    # model.compile(optimizer=Adam(learning_rate=LR), loss='categorical_crossentropy',
    #                metrics=['accuracy', f1_metric] )
    ## if you wanna use scheduler for learning rate:
    ## set learning_rate=lr_schedule
    ## change the patience on early stopping to 10, so that it can train longer
    ## don't use the reduce_lr callback, it is not needed

    model.compile(optimizer=AdamW(learning_rate=LR, weight_decay=0.01), loss='categorical_crossentropy',
                   metrics=['accuracy', f1_metric])
    print(model.summary())
    
    # model_checkpoint = ModelCheckpoint(filepath=os.path.join(OUTPUT_DIR, f'{model_name}_best.h5'),
    #                                     monitor='val_accuracy', save_best_only=True, verbose=1)
    print('--'* 50)
    print(f'Using {DATASET_PERC*100}% of the dataset for training ({int(df_full.shape[0]*DATASET_PERC)} rows) .')
    print(f'Starting CNN training with parameters:')
    print(f'------>number of epochs={N_EPOCHS}, batch_size={BATCH_SIZE}, initial learning_rate={LR}') 
    print(f'------>embedding_dim={EMBEDDING_DIM}, max_sequence_lenght={MAX_SEQUENCE_LENGTH}, vocab_limit={VOCAB_LIMIT}')
    print(f'------>conv_filters={CONV_FILTERS}, conv_k_list={CONV_K_LIST}, conv_k_reg={CONV_K_REG}, dense_k_reg={DENSE_K_REG}')
    print('--'* 50)

    history = model.fit(
        X_train, y_train,
        class_weight=class_weight_dict, # apply class weights
        batch_size=BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[early_stopping, reduce_lr],
        #callbacks=[early_stopping],
        verbose=1)
    
    print("CNN training completed!")
    max_f1_score = np.max(history.history['val_f1_score']).round(3)
    model_name = f'text-cnn-lr-{LR}-f1-{max_f1_score}'

    if max_f1_score > 0.70:
        print(f"Validation F1 score is high enough: {max_f1_score}")
    
        model.save(os.path.join(OUT_PATH, f'{model_name}.keras'))
        print(f"Model saved as {model_name}_final.keras")

            # Save training history
        history_dict = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss'],
            'accuracy': history.history['accuracy'],
            'val_accuracy': history.history['val_accuracy'],
            'f1_score': history.history['f1_score'],  # Training F1 score
            'val_f1_score': history.history['val_f1_score'],  # Validation F1 score
            'learning_rate': history.history.get('learning_rate', [LR] * len(history.history['loss'])),
        }
    
        with open(os.path.join(OUT_PATH, f'{model_name}_history.json'), 'w') as f:
            json.dump(history_dict, f, indent=2)
        print("Generating evaluation report...")

        # Run SHAP analysis
        cat_mapping = df[['prdtypecode', 'cat_name_en']].drop_duplicates()
        sorted_codes = sorted(cat_mapping['prdtypecode'].unique())
        class_names = {i: cat_mapping[cat_mapping['prdtypecode'] == code]['cat_name_en'].iloc[0] 
               for i, code in enumerate(sorted_codes)}
        y_val_original = y_val.copy()
        importance_analysis(model, X_train, X_val, y_val_original, vocab_filtered, 
                     os.path.join(OUT_PATH, f"importance_analysis_{model_name}.txt"), 
                     samples_per_class=15, class_names=class_names)

    
    summary, report, predictions = get_training_report(
        model=model,
        history=history, 
        validation_data=(X_val, y_val),
        save_dir=OUT_PATH,
        model_name=model_name,
        h_parameters=hyperparams_dict)
    
    np.save(os.path.join(OUT_PATH, f'{model_name}_predictions.npy'), predictions)
    return model, history

if __name__ == "__main__":
    t1 = time.time()
    model, history = main()
    t2 = time.time()
    print(f"Total time taken: {(t2 - t1) / 60:.2f} minutes")