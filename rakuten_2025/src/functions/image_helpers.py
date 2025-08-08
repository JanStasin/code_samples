import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import io

def load_and_preprocess_image(image_path, target_size=(224, 224)):
    try:
        # Load image
        img = load_img(image_path, target_size=target_size)
        img_array = img_to_array(img)
        # Normalize pixel values to [0, 1]
        #img_array = img_array / 255.0
        img_array = (img_array / 127.5) - 1.0
        return img_array
    except Exception as e:
        print(f'Error loading image {image_path}: {e}')
        # Return zeros if image can't be loaded
        return np.zeros((*target_size, 3))

def load_images_batch(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = load_and_preprocess_image(path, target_size)
        images.append(img)
    return np.array(images)


def get_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            w, h = img.size
            num_pixels = w * h
            return w, h, num_pixels
    except:
        return None, None, None
    
def get_blur_score(image_path, thr=100):
    ''' get blur score and check if image is sharp enough (True/False)'''
    try:
        img = cv2.imread(image_path, 0)  # grayscale
        if img is None:
            return False
        blur_score = cv2.Laplacian(img, cv2.CV_64F).var()
            
        return blur_score, blur_score > thr
    except:
        return None, False

def get_brightness(image_path, thr=50.):
    '''Get average brightness (0-255)'''
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            pixels = np.array(img)
            brightness = pixels.mean()
            brightness = round(brightness, 1)
            return brightness, brightness > thr
    except:
        return None, False
    
def get_rgb(image_path):
    try:
       with Image.open(image_path) as img:
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')

            pixels = np.array(img)
            r, g, b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
            r_avg, g_avg, b_avg = r.mean(), g.mean(), b.mean()
            contrast = round((r.std() + g.std() + b.std()) / 3, 2)

            # Classify color type
            if r_avg < 5 and g_avg < 5 and b_avg < 5:
                color_type = 'full_black'
            elif max(abs(r_avg - g_avg), abs(r_avg - b_avg), abs(g_avg - b_avg)) < 10:
                color_type = 'greyscale'
            else:
                color_type = 'color'
           
            return (r_avg, g_avg, b_avg), color_type, contrast
    except:
        return None , None, None
    
def get_summary(results_df):
    '''Get basic summary statistics'''
    existing = results_df[results_df['exists'] == True]
    
    return {
        'total_images': len(results_df),
        'missing_images': len(results_df) - len(existing),
        # 'avg_width': existing['width'].mean() if len(existing) > 0 else 0,
        # 'avg_height': existing['height'].mean() if len(existing) > 0 else 0,
        'avg_sharpness': existing['sharpness'].mean() if len(existing) > 0 else 0,
        'sharp_images': (existing['is_sharp'].sum() if len(existing) > 0 else 0)/ len(existing),
        'avg_brightness': existing['brightness'].mean() if len(existing) > 0 else 0,
        'bright_images': (existing['is_bright'].sum() if len(existing) > 0 else 0)/ len(existing),
        'contrast': existing['contrast'].mean() if len(existing) > 0 else 0,
        'color_types': existing['color_type'].value_counts().to_dict(),
        #'avg_rgb': existing['average_rgb'].apply(lambda x: x if isinstance(x, tuple) else (None, None, None)).tolist()    
        }



import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from rembg import remove
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import random
from rembg import remove
import warnings
warnings.filterwarnings('ignore')

# CELL 1: Setup and Baseline Processing (Minimal: Resize + Normalize)
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

# Updated processing functions that maintain consistent image sets
def process_all_methods_synchronized(df_img, n_samples=1000):
    """Process same images with all three methods to ensure consistency"""
    sample_df = df_img.sample(n=min(n_samples, len(df_img))).reset_index(drop=True)
    
    baseline_imgs = []
    bg_removed_imgs = []
    smart_crop_imgs = []
    successful_paths = []
    
    for idx, row in sample_df.iterrows():
        image_path = row['imagepath_orig']  # Use original images
        
        # Try all three preprocessing methods
        baseline_result = baseline_preprocessing(image_path)
        bg_removal_result = background_removal_preprocessing(image_path) 
        smart_crop_result = smart_crop_preprocessing(image_path)
        
        # Only keep images that succeeded in ALL methods
        if all([baseline_result is not None, bg_removal_result is not None, smart_crop_result is not None]):
            baseline_imgs.append(baseline_result)
            bg_removed_imgs.append(bg_removal_result)
            smart_crop_imgs.append(smart_crop_result)
            successful_paths.append(image_path)
        
        if idx % 50 == 0:
            print(f"Synchronized processing: {idx+1}/{len(sample_df)}, Successful: {len(successful_paths)}")
    
    print(f"Successfully processed {len(successful_paths)} images with all methods")
    return successful_paths, baseline_imgs, bg_removed_imgs, smart_crop_imgs

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

# # Updated usage example:
# N = 1000  # Number of images to process

# # Process all methods synchronously to ensure same images
# paths, baseline_imgs, bg_removed_imgs, smart_crop_imgs = process_all_methods_synchronized(df_img, N)

# # Display comparison - now all methods will show the same original images
# display_preprocessing_comparison(paths, baseline_imgs, bg_removed_imgs, smart_crop_imgs)