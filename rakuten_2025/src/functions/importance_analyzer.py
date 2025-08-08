import numpy as np
import json


def importance_analysis(model, X_train, X_val, y_val, vocab_dict, 
                           output_file='token_importance_analysis.txt', 
                           samples_per_class=10,
                           class_names=None):
    """
    Analyze token importance using occlusion (masking) method.
    
    Args:
        model: Trained CNN model
        X_train: Training data (not used here)
        X_val: Validation sequences
        y_val: Validation labels (not one-hot)
        vocab_dict: Token to index mapping
        output_file: Where to save results
        samples_per_class: How many samples to analyze per class
        class_names: Optional dict mapping class indices to names
    """
    
    print('Setting up token importance analysis using occlusion method...')
   
    print(f'Debug - X_val shape: {X_val.shape}, dtype: {X_val.dtype}')
    
    # Ensure y_val is 1D array of integers
    if len(y_val.shape) > 1:
        # If one-hot encoded, convert back
        y_val = np.argmax(y_val, axis=1)
    y_val = y_val.astype(int)
    
    # Create reverse vocabulary
    reverse_vocab = {v: k for k, v in vocab_dict.items()}
    
    # Collect stratified samples
    results = []
    analyzed_count = 0
    
    unique_classes = np.unique(y_val)
    print(f'Found {len(unique_classes)} unique classes')
    
    for class_id in unique_classes:
        class_indices = np.where(y_val == class_id)[0]
        n_samples = min(samples_per_class, len(class_indices))
        selected_indices = np.random.choice(class_indices, n_samples, replace=False)
        
        for idx in selected_indices:
            try:
                # Get base prediction
                X_sample = X_val[idx:idx+1]
                base_probs = model.predict(X_sample, verbose=0)[0]
                pred_class = np.argmax(base_probs)
                base_confidence = base_probs[pred_class]
                
                # Calculate importance by masking each token
                token_importance = []
                sequence = X_val[idx]  # Get the actual sequence
                
                for i in range(len(sequence)):
                    token_id = sequence[i]
                    if token_id > 0:  # Skip padding
                        # Create masked version
                        X_masked = X_sample.copy()
                        X_masked[0, i] = 0  # Mask this token
                        
                        # Get new prediction
                        masked_probs = model.predict(X_masked, verbose=0)[0]
                        masked_confidence = masked_probs[pred_class]
                        
                        # Importance = drop in confidence when masked
                        importance = float(base_confidence - masked_confidence)
                        token = reverse_vocab.get(int(token_id), f'UNK_{token_id}')
                        token_importance.append((token, importance))
                
                # Get top 5 tokens
                if token_importance:
                    token_importance.sort(key=lambda x: x[1], reverse=True)
                    top_tokens = [t[0] for t in token_importance[:5]]
                else:
                    top_tokens = ['<empty>']
                
                # Store result - ensure all values are Python native types
                results.append({
                    'sample_id': int(idx),
                    'predicted': int(pred_class),
                    'confidence': float(base_confidence),
                    'true': int(y_val[idx]),
                    'correct': bool(int(pred_class) == int(y_val[idx])),
                    'top_tokens': top_tokens
                })
                
                analyzed_count += 1
                if analyzed_count % 15 == 0:
                    print(f'Analyzed {analyzed_count} samples...')
                    
            except Exception as e:
                print(f'Error analyzing sample {idx}: {e}')
                print(f'Debug - idx type: {type(idx)}, value: {idx}')
                print(f'Debug - y_val[idx] type: {type(y_val[idx])}, value: {y_val[idx]}')
                continue
    
    print(f'Successfully analyzed {len(results)} samples')
    
    # Save results
    with open(output_file, 'w') as f:
        f.write('Token Importance Analysis Results (Occlusion Method)\n')
        f.write('=' * 60 + '\n\n')
        
        for r in results:
            f.write(f"Sample ID: {r['sample_id']}\n")
            
            pred_name = class_names.get(r['predicted'], f"Class_{r['predicted']}") if class_names else f"Class_{r['predicted']}"
            true_name = class_names.get(r['true'], f"Class_{r['true']}") if class_names else f"Class_{r['true']}"
            
            f.write(f"Predicted: {pred_name} (confidence: {r['confidence']:.2f})\n")
            f.write(f"True: {true_name} (Correct: {'Yes' if r['correct'] else 'No'})\n")
            f.write(f"Top contributing tokens: {r['top_tokens']}\n")
            f.write('-' * 60 + '\n\n')
        
        if results:
            accuracy = sum(r['correct'] for r in results) / len(results)
            f.write(f'\nAccuracy: {accuracy:.1%} ({len(results)} samples)\n')
    
    # Save JSON
    with open(output_file.replace('.txt', '.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f'Results saved to: {output_file}')
    return results

