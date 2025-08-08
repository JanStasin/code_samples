import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import json
import os
from datetime import datetime
from matplotlib.backends.backend_pdf import PdfPages
from tensorflow.keras.utils import Sequence

def get_training_report(model, history, validation_data, class_names=None, save_dir='../reports', model_name='model', h_parameters=None):
    """
    Flexible model evaluation report generator that works with both generators and arrays
    
    Args:
    - model: trained Keras model
    - history: training history object (from model.fit())
    - validation_data: Either:
        * Keras generator (ImageDataGenerator, etc.)
        * Tuple of (X_val, y_val) numpy arrays
    - class_names: list of class names (optional)
    - save_dir: directory to save reports
    - model_name: name for saving files
    - h_parameters: dictionary or string of hyperparameters (optional)
    """
    
    # Create save directory
    timestamp = datetime.now().strftime("%m%d_%H%M")
    os.makedirs(save_dir, exist_ok=True)

    # Get final validation F1 score for filename (with error handling)
    try:
        if 'val_f1_score' in history.history:
            final_val_f1_for_filename = f"{history.history['val_f1_score'][-1]:.4f}"
        else:
            final_val_f1_for_filename = "na"
    except (KeyError, IndexError):
        final_val_f1_for_filename = "na"

    # Create PDF filename
    pdf_filename = f'{save_dir}/training_{model_name}_f1-{final_val_f1_for_filename}_t-{timestamp}.pdf'
    
    print("Evaluating model on validation set...")
    
    # Detect data type and extract information
    is_generator = hasattr(validation_data, 'class_indices') or isinstance(validation_data, Sequence)
    
    if is_generator:
        # Generator case (images)
        generator = validation_data
        
        # Use Keras built-in evaluate
        val_metrics = model.evaluate(generator, verbose=1)
        metric_names = model.metrics_names
        val_results = dict(zip(metric_names, val_metrics))
        
        # Generate predictions
        print("Generating predictions...")
        generator.reset()
        y_pred = model.predict(generator, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = generator.classes
        
        # Get class names and metadata
        if class_names is None:
            class_names = list(generator.class_indices.keys())
        
        batch_size = generator.batch_size
        input_info = f"Image Size: {generator.target_size}"
        data_type = "Image Generator"
        
    else:
        # Array case (text, tabular, etc.)
        X_val, y_val = validation_data
        
        # Use Keras built-in evaluate
        val_metrics = model.evaluate(X_val, y_val, verbose=1)
        metric_names = model.metrics_names
        val_results = dict(zip(metric_names, val_metrics))
        
        # Generate predictions
        print("Generating predictions...")
        y_pred = model.predict(X_val, verbose=1)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Handle y_val format (one-hot or integer)
        if len(y_val.shape) > 1 and y_val.shape[1] > 1:
            # One-hot encoded
            y_true = np.argmax(y_val, axis=1)
            num_classes = y_val.shape[1]
        else:
            # Integer labels
            y_true = y_val
            num_classes = len(np.unique(y_true))
        
        # Generate class names if not provided
        if class_names is None:
            class_names = [f'Class_{i}' for i in range(num_classes)]
        
        batch_size = "N/A (arrays)"
        input_info = f"Input Shape: {X_val.shape}"
        data_type = "Numpy Arrays"
    
    # Calculate additional metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    if 'f1_score' in history.history:
        final_train_f1 = history.history['f1_score'][-1]
        final_val_f1 = history.history['val_f1_score'][-1]
    else:
        final_train_f1 = final_val_f1 = "N/A"
    
    # Generate classification report
    class_report = classification_report(y_true, y_pred_classes, 
                                       target_names=class_names, 
                                       output_dict=True)
    
    # Create compact 2-page PDF
    with PdfPages(pdf_filename) as pdf:
        
        # PAGE 1: TRAINING CURVES + SUMMARY
        fig = plt.figure(figsize=(11, 8.5))
        
        # Create grid: 2x3 layout
        gs = fig.add_gridspec(2, 3, height_ratios=[1, 1], width_ratios=[1, 1, 0.8])
        
        # Training curves (top row)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(history.history['loss'], label='Train', linewidth=2)
        ax1.plot(history.history['val_loss'], label='Val', linewidth=2)
        ax1.set_title('Loss', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(history.history['accuracy'], label='Train', linewidth=2)
        ax2.plot(history.history['val_accuracy'], label='Val', linewidth=2)
        ax2.set_title('Accuracy', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # F1 and LR (bottom row)
        ax3 = fig.add_subplot(gs[1, 0])
        if 'f1_score' in history.history:
            ax3.plot(history.history['f1_score'], label='Train', linewidth=2)
            ax3.plot(history.history['val_f1_score'], label='Val', linewidth=2)
            ax3.set_title('F1 Score', fontweight='bold')
            ax3.legend()
        else:
            ax3.text(0.5, 0.5, 'F1 not tracked', ha='center', va='center')
            ax3.set_title('F1 Score (N/A)')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(gs[1, 1])
        if 'learning_rate' in history.history:
            ax4.plot(history.history['learning_rate'], linewidth=2, color='red')
            ax4.set_title('Learning Rate', fontweight='bold')
            ax4.set_yscale('log')
        else:
            ax4.text(0.5, 0.5, 'LR not tracked', ha='center', va='center')
            ax4.set_title('Learning Rate (N/A)')
        ax4.grid(True, alpha=0.3)
        
        # Summary text (right side)
        ax_text = fig.add_subplot(gs[:, 2])
        ax_text.axis('off')
        
        # Format F1 scores safely
        train_f1_str = f"{final_train_f1:.3f}" if final_train_f1 != "N/A" else "N/A"
        val_f1_str = f"{final_val_f1:.3f}" if final_val_f1 != "N/A" else "N/A"
        
        # Format hyperparameters for display
        hyperparams_text = ""
        if h_parameters and h_parameters != 'None':
            if isinstance(h_parameters, dict):
                hyperparams_text = "\n\nHYPERPARAMS\n"
                for key, value in h_parameters.items():
                    # Truncate long values for display
                    str_value = str(value)
                    if len(str_value) > 15:
                        str_value = str_value[:12] + "..."
                    hyperparams_text += f"{key}: {str_value}\n"
            else:
                hyperparams_text = f"\n\nHYPERPARAMS\n{h_parameters}\n"
        
        summary_text = f"""{model_name}
{datetime.now().strftime('%Y-%m-%d %H:%M')}
Data: {data_type}

FINAL METRICS
Train Acc: {final_train_acc:.3f}
Val Acc:   {final_val_acc:.3f}
Train F1:  {train_f1_str}
Val F1:    {val_f1_str}

Macro F1:  {class_report['macro avg']['f1-score']:.3f}
Weight F1: {class_report['weighted avg']['f1-score']:.3f}

MODEL INFO
Params: {model.count_params():,}
Epochs: {len(history.history['loss'])}
Classes: {len(class_names)}
{input_info}{hyperparams_text}"""
        
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes, 
                    fontsize=9, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Training Report - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
        # PAGE 2: CONFUSION MATRIX + CLASS METRICS
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1])
        
        # Confusion matrix (top, spanning both columns)
        ax_cm = fig.add_subplot(gs[0, :])
        cm = confusion_matrix(y_true, y_pred_classes)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names, ax=ax_cm)
        ax_cm.set_title('Confusion Matrix', fontweight='bold')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('Actual')
        
        # Class metrics (bottom row)
        report_df = pd.DataFrame(class_report).transpose()
        class_metrics = report_df.loc[class_names, ['precision', 'recall', 'f1-score']]
        
        # Top 10 and bottom 10 classes by F1
        sorted_by_f1 = class_metrics.sort_values('f1-score')
        worst_10 = sorted_by_f1.head(10)
        best_10 = sorted_by_f1.tail(10)
        
        ax_worst = fig.add_subplot(gs[1, 0])
        worst_10['f1-score'].plot(kind='barh', ax=ax_worst, color='lightcoral')
        ax_worst.set_title('Bottom 10 Classes (F1 Score)', fontweight='bold')
        ax_worst.set_xlabel('F1 Score')
        
        ax_best = fig.add_subplot(gs[1, 1])
        best_10['f1-score'].plot(kind='barh', ax=ax_best, color='lightgreen')
        ax_best.set_title('Top 10 Classes (F1 Score)', fontweight='bold')
        ax_best.set_xlabel('F1 Score')
        
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
    
    # Also save JSON files
    summary = {
        'Model Name': model_name,
        'Timestamp': timestamp,
        'Data Type': data_type,
        'Total Parameters': model.count_params(),
        'Training Epochs': len(history.history['loss']),
        'Final Training Accuracy': f"{final_train_acc:.4f}",
        'Final Validation Accuracy': f"{final_val_acc:.4f}",
        'Final Training Loss': f"{final_train_loss:.4f}",
        'Final Validation Loss': f"{final_val_loss:.4f}",
        'Final Training F1': f"{final_train_f1:.4f}" if final_train_f1 != "N/A" else "N/A",
        'Final Validation F1': f"{final_val_f1:.4f}" if final_val_f1 != "N/A" else "N/A",
        'Validation Metrics': val_results,
        'Macro Avg F1': f"{class_report['macro avg']['f1-score']:.4f}",
        'Weighted Avg F1': f"{class_report['weighted avg']['f1-score']:.4f}",
        'Number of Classes': len(class_names),
        'Batch Size': batch_size,
        'Input Info': input_info,
        'Hyperparameters': h_parameters if h_parameters and h_parameters != 'None' else {}
    }
    
    with open(f'{save_dir}/summary_{model_name}_{timestamp}.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nPDF report saved as: {pdf_filename}")
    print(f"JSON summary saved as: {save_dir}/summary_{model_name}_{timestamp}.json")
    
    return summary, class_report, y_pred

# USAGE EXAMPLES:

#
# summary, report = get_training_report(
#     model=model,
#     history=history, 
#     validation_data=(X_val, y_val),
#     h_parameters=hyperparams_dict,
#     save_dir='../reports/',
#     model_name='text_cnn_v1'
# )

