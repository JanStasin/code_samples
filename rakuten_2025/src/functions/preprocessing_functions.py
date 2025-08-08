import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from rembg import remove
import warnings
warnings.filterwarnings('ignore')
import io
import os

# CELL 1: Setup and Advanced Processing Methods
def advanced_augmentation_preprocessing(image_path, target_size=(500, 500)):
    """Advanced preprocessing with augmentation and enhancement"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio with padding
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create white background and center the image
        result = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Advanced enhancements
        # 1. Denoising
        result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
        
        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        lab[:,:,0] = clahe.apply(lab[:,:,0])
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # 3. Sharpening kernel
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        result = cv2.filter2D(result, -1, kernel)
        
        # 4. Color balance adjustment
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=10)
        
        # Normalize to [0, 1]
        result = result.astype(np.float32) / 255.0
        
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def baseline_preprocessing(image_path, target_size=(500, 500)):
    """Baseline: Simple resize and normalize"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize maintaining aspect ratio with padding
        h, w = img.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Create white background and center the image
        result = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Normalize to [0, 1]
        result = result.astype(np.float32) / 255.0
        
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_baseline_batch(df_img, n_samples=1000):
    """Process N images with baseline preprocessing"""
    sample_df = df_img.sample(n=min(n_samples, len(df_img))).reset_index(drop=True)
    processed_images = []
    original_paths = []
    
    for idx, row in sample_df.iterrows():
        processed = baseline_preprocessing(row['image_path'])
        if processed is not None:
            processed_images.append(processed)
            original_paths.append(row['image_path'])
        
        if idx % 100 == 0:
            print(f"Baseline processing: {idx+1}/{len(sample_df)}")
    
    return processed_images, original_paths

# CELL 2: Background Removal + Standardization
def background_removal_preprocessing(image_path, target_size=(500, 500)):
    """Background removal with rembg + standardization"""
    try:
        # Load image
        with open(image_path, 'rb') as f:
            input_image = f.read()
        
        # Remove background
        output_image = remove(input_image)
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(output_image)).convert('RGBA')
        
        # Create white background
        background = Image.new('RGB', img.size, (255, 255, 255))
        
        # Paste the image with transparency
        background.paste(img, mask=img.split()[-1])
        
        # Convert to numpy array
        img_array = np.array(background)
        
        # Resize maintaining aspect ratio
        h, w = img_array.shape[:2]
        scale = min(target_size[0]/w, target_size[1]/h)
        new_w, new_h = int(w * scale), int(h * scale)
        
        resized = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # Center on white background
        result = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 255
        y_offset = (target_size[1] - new_h) // 2
        x_offset = (target_size[0] - new_w) // 2
        result[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        
        # Additional standardization
        # Enhance contrast slightly
        result = cv2.convertScaleAbs(result, alpha=1.1, beta=5)
        
        # Normalize
        result = result.astype(np.float32) / 255.0
        
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_background_removal_batch(df_img, n_samples=1000):
    """Process N images with background removal"""
    sample_df = df_img.sample(n=min(n_samples, len(df_img))).reset_index(drop=True)
    processed_images = []
    original_paths = []
    
    for idx, row in sample_df.iterrows():
        processed = background_removal_preprocessing(row['image_path'])
        if processed is not None:
            processed_images.append(processed)
            original_paths.append(row['image_path'])
        
        if idx % 50 == 0:  # Less frequent updates due to slower processing
            print(f"Background removal processing: {idx+1}/{len(sample_df)}")
    
    return processed_images, original_paths

# CELL 3: Smart Cropping + Standardization
def smart_crop_preprocessing(image_path, target_size=(500, 500)):
    """Smart cropping to product boundaries + standardization"""
    try:
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img.copy()
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find the largest contour (likely the product)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Add padding (10% of image dimensions)
            padding_x = int(0.1 * img.shape[1])
            padding_y = int(0.1 * img.shape[0])
            
            x = max(0, x - padding_x)
            y = max(0, y - padding_y)
            w = min(img.shape[1] - x, w + 2 * padding_x)
            h = min(img.shape[0] - y, h + 2 * padding_y)
            
            # Crop to bounding box
            cropped = img[y:y+h, x:x+w]
        else:
            # Fallback: use center crop
            h, w = img.shape[:2]
            crop_size = min(h, w)
            start_x = (w - crop_size) // 2
            start_y = (h - crop_size) // 2
            cropped = img[start_y:start_y+crop_size, start_x:start_x+crop_size]
        
        # Resize to target size
        resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
        
        # Standardization: Histogram equalization for better contrast
        # Convert to LAB color space
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        lab[:,:,0] = cv2.equalizeHist(lab[:,:,0])  # Equalize L channel
        result = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        # Normalize
        result = result.astype(np.float32) / 255.0
        
        return result
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process_smart_crop_batch(df_img, n_samples=1000):
    """Process N images with smart cropping"""
    sample_df = df_img.sample(n=min(n_samples, len(df_img))).reset_index(drop=True)
    processed_images = []
    original_paths = []
    
    for idx, row in sample_df.iterrows():
        processed = smart_crop_preprocessing(row['image_path'])
        if processed is not None:
            processed_images.append(processed)
            original_paths.append(row['image_path'])
        
        if idx % 100 == 0:
            print(f"Smart crop processing: {idx+1}/{len(sample_df)}")
    
    return processed_images, original_paths

# Memory-efficient processing with saving capability
def process_all_methods_memory_efficient(df_img, n_samples=1000, save_processed=False, save_dir=None):
    """
    Memory-efficient processing that saves images and only keeps a sample in memory
    
    Args:
        df_img: DataFrame with image paths
        n_samples: Number of images to process
        save_processed: Whether to save processed images to disk
        save_dir: Directory to save processed images (if save_processed=True)
    
    Returns:
        paths, sample_baseline, sample_bg_removed, sample_smart_crop, sample_advanced
        (only returns small sample for visualization, but processes all requested images)
    """
    sample_df = df_img.sample(n=min(n_samples, len(df_img))).reset_index(drop=True)
    
    # Create save directories if needed
    if save_processed and save_dir:
        os.makedirs(os.path.join(save_dir, 'baseline'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'background_removed'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'smart_crop'), exist_ok=True)
        os.makedirs(os.path.join(save_dir, 'advanced'), exist_ok=True)
    
    # For memory efficiency, only keep a small sample for visualization
    max_memory_samples = 50
    sample_baseline = []
    sample_bg_removed = []
    sample_smart_crop = []
    sample_advanced = []
    successful_paths = []
    
    processing_methods = {
        'baseline': baseline_preprocessing,
        'background_removed': background_removal_preprocessing,
        'smart_crop': smart_crop_preprocessing,
        'advanced': advanced_augmentation_preprocessing
    }
    
    successful_count = 0
    
    for idx, row in sample_df.iterrows():
        image_path = row['imagepath_orig']
        
        # Get base filename for saving
        base_filename = os.path.splitext(os.path.basename(image_path))[0]
        
        # Process with all methods
        results = {}
        all_successful = True
        
        for method_name, method_func in processing_methods.items():
            result = method_func(image_path)
            if result is None:
                all_successful = False
                break
            results[method_name] = result
        
        # Only proceed if all methods succeeded
        if all_successful:
            successful_count += 1
            
            # Save processed images if requested
            if save_processed and save_dir:
                for method_name, processed_img in results.items():
                    # Convert normalized image back to uint8 for saving
                    img_to_save = (processed_img * 255).astype(np.uint8)
                    img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                    
                    save_path = os.path.join(save_dir, method_name, f"{base_filename}_processed.png")
                    cv2.imwrite(save_path, img_to_save)
            
            # Keep only a limited number in memory for visualization
            if len(sample_baseline) < max_memory_samples:
                sample_baseline.append(results['baseline'])
                sample_bg_removed.append(results['background_removed'])
                sample_smart_crop.append(results['smart_crop'])
                sample_advanced.append(results['advanced'])
                successful_paths.append(image_path)
        
        if idx % 50 == 0:
            print(f"Processed: {idx+1}/{len(sample_df)}, Successful: {successful_count}")
    
    print(f"Successfully processed {successful_count} images with all methods")
    if save_processed and save_dir:
        print(f"Processed images saved to: {save_dir}")
    
    return successful_paths, sample_baseline, sample_bg_removed, sample_smart_crop, sample_advanced

# Updated visualization function for 4 methods
def display_preprocessing_comparison_4methods(image_paths, baseline_imgs, bg_removed_imgs, smart_crop_imgs, advanced_imgs, n_display=4):
    """Display 4x5 grid comparing all preprocessing approaches"""
    
    if len(image_paths) < n_display:
        n_display = len(image_paths)
    
    indices = random.sample(range(len(image_paths)), n_display)
    
    fig, axes = plt.subplots(n_display, 5, figsize=(20, 4*n_display))
    if n_display == 1:
        axes = axes.reshape(1, -1)
    
    methods = ['Original', 'Baseline', 'Background Removed', 'Smart Crop', 'Advanced']
    
    for i, idx in enumerate(indices):
        # Load original image
        original = cv2.imread(image_paths[idx])
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        images = [
            original,
            baseline_imgs[idx],
            bg_removed_imgs[idx],
            smart_crop_imgs[idx],
            advanced_imgs[idx]
        ]
        
        for j, (img, method) in enumerate(zip(images, methods)):
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{method}' + (f'\nImage {i+1}' if j == 0 else ''))
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Batch processing function for large datasets
def process_entire_dataset(df_img, save_dir, batch_size=100, method='baseline'):
    """
    Process entire dataset in batches without keeping images in memory
    
    Args:
        df_img: DataFrame with image paths
        save_dir: Directory to save processed images
        batch_size: Number of images to process per batch
        method: Which preprocessing method to use
    """
    os.makedirs(save_dir, exist_ok=True)
    
    method_functions = {
        'baseline': baseline_preprocessing,
        'background_removed': background_removal_preprocessing,
        'smart_crop': smart_crop_preprocessing,
        'advanced': advanced_augmentation_preprocessing
    }
    
    if method not in method_functions:
        raise ValueError(f"Method must be one of: {list(method_functions.keys())}")
    
    processing_func = method_functions[method]
    total_images = len(df_img)
    processed_count = 0
    failed_count = 0
    
    for start_idx in range(0, total_images, batch_size):
        end_idx = min(start_idx + batch_size, total_images)
        batch = df_img.iloc[start_idx:end_idx]
        
        print(f"Processing batch {start_idx//batch_size + 1}: images {start_idx+1}-{end_idx}")
        
        for _, row in batch.iterrows():
            image_path = row['imagepath_orig']
            base_filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Process image
            processed_img = processing_func(image_path)
            
            if processed_img is not None:
                # Convert and save
                img_to_save = (processed_img * 255).astype(np.uint8)
                img_to_save = cv2.cvtColor(img_to_save, cv2.COLOR_RGB2BGR)
                
                save_path = os.path.join(save_dir, f"{base_filename}_{method}.png")
                cv2.imwrite(save_path, img_to_save)
                processed_count += 1
            else:
                failed_count += 1
        
        # Print progress
        print(f"Batch complete. Total processed: {processed_count}, Failed: {failed_count}")
    
    print(f"Dataset processing complete!")
    print(f"Successfully processed: {processed_count}/{total_images} images")
    print(f"Failed: {failed_count} images")
    print(f"Processed images saved to: {save_dir}")

# Visualization function
def display_preprocessing_comparison(image_paths, baseline_imgs, bg_removed_imgs, smart_crop_imgs, n_display=4):
    """Display 4x4 grid comparing preprocessing approaches"""
    
    if len(image_paths) < n_display:
        n_display = len(image_paths)
    
    indices = random.sample(range(len(image_paths)), n_display)
    
    fig, axes = plt.subplots(n_display, 4, figsize=(16, 4*n_display))
    if n_display == 1:
        axes = axes.reshape(1, -1)
    
    methods = ['Original', 'Baseline', 'Background Removed', 'Smart Crop']
    
    for i, idx in enumerate(indices):
        # Load original image
        original = cv2.imread(image_paths[idx])
        original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
        
        images = [
            original,
            baseline_imgs[idx],
            bg_removed_imgs[idx],
            smart_crop_imgs[idx]
        ]
        
        for j, (img, method) in enumerate(zip(images, methods)):
            axes[i, j].imshow(img)
            axes[i, j].set_title(f'{method}' + (f'\nImage {i+1}' if j == 0 else ''))
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage:
# N = 1000  # Number of images to process
# baseline_imgs, paths1 = process_baseline_batch(df_img, N)
# bg_removed_imgs, paths2 = process_background_removal_batch(df_img, N)
# smart_crop_imgs, paths3 = process_smart_crop_batch(df_img, N)
# display_preprocessing_comparison(paths1, baseline_imgs, bg_removed_imgs, smart_crop_imgs)