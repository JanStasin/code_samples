import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
import os
import imageio
from dotenv import load_dotenv
# Load environment variables from .env file
# By default, load_dotenv() looks for .env in the current directory or parent directories.
load_dotenv()

image_file_extension = 'png'
IMAGE_SIZE = (224, 224)
# Load a pre-trained CNN for feature extraction
base_image_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    include_top=False,
    weights='imagenet'
)
base_image_model.trainable = False # Freeze the base model to use it as a feature extractor

def load_and_preprocess_image(image_path):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(IMAGE_SIZE)
        img_array = np.array(img).astype('float32') / 255.0
        # print(f"Success: Image found at {image_path}")
        return img_array
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Returning a dummy black image.")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32) # Return black dummy image
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Returning a dummy black image.")
        return np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32)

        # Split into R, G, B, A
        r, g, b, a = rgba_img.split()

        # Convert RGB part to numpy array and normalize
        rgb_img_array = np.array(Image.merge('RGB', (r, g, b))).astype('float32') / 255.0

        # Convert Alpha part to numpy array and normalize
        alpha_channel_array = np.array(a).astype('float32') / 255.0
        # Ensure alpha has a channel dimension (H, W, 1) for consistency in Keras layers
        alpha_channel_array = np.expand_dims(alpha_channel_array, axis=-1)
        # print(f"Success: Image found at {image_path}")
        return rgb_img_array, alpha_channel_array
    except FileNotFoundError:
        print(f"Warning: Image not found at {image_path}. Returning dummy RGB and Alpha.")
        return (np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32),
                np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32))
    except Exception as e:
        print(f"Error loading image {image_path}: {e}. Returning dummy RGB and Alpha.")
        return (np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.float32),
                np.zeros((IMAGE_SIZE[0], IMAGE_SIZE[1], 1), dtype=np.float32))

def get_image_full_path(image_id, product_id, base_folder, file_extension = image_file_extension):
    return os.path.join(base_folder, f'image_{image_id}_product_{product_id}.{file_extension}')


#############################################################

# --- 1. Helper Function to Load and Preprocess Original Image for Grad-CAM ---
def load_and_preprocess_original_image_for_gradcam(image_path, target_size=IMAGE_SIZE):
    """
    Loads and preprocesses an image (PNG with alpha channel) for direct input
    into the base_image_model, explicitly masking out transparent areas to black.
    Returns a TF Tensor with batch dimension (1, H, W, 3).
    """
    img_raw = tf.io.read_file(image_path)
    img_rgba = tf.image.decode_image(img_raw, channels=4, expand_animations=False)
    img_rgba = tf.image.resize(img_rgba, target_size)
    img_rgba = tf.cast(img_rgba, tf.float32) / 255.0

    rgb_channels = img_rgba[..., :3]
    alpha_channel = img_rgba[..., 3:]

    masked_rgb_image = rgb_channels * alpha_channel

    img_rgb_final = tf.expand_dims(masked_rgb_image, axis=0)
    return img_rgb_final


# --- 2. Function to Generate Grad-CAM Heatmap for Multimodal Model ---
def make_gradcam_heatmap(
    img_array_raw, # Raw image (1, H, W, 3) for base_image_model (already masked)
    multimodal_model, # Your full trained model
    base_image_model, # The MobileNetV2 part
    target_conv_layer_name, # The specific convolutional layer to explain
    tfidf_feat_batch, # (1, TFIDF_DIM)
    alpha_feat_batch, # (1, 1)
    pred_index=None # Index of the class to explain (defaults to highest prediction)
):
    """
    Generates a Grad-CAM heatmap for a given image and predicted class
    within a multimodal model, targeting a specific convolutional layer.
    """
    target_layer = base_image_model.get_layer(target_conv_layer_name)

    grad_model_image_branch = Model(
        inputs=base_image_model.input,
        outputs=[target_layer.output, GlobalAveragePooling2D()(base_image_model.output)]
    )

    with tf.GradientTape() as tape:
        target_layer_output, pooled_rgb_features = grad_model_image_branch(img_array_raw)
        tape.watch(target_layer_output)

        model_inputs_for_prediction = {
            'text_input_multi_tfidf': tfidf_feat_batch,
            'image_input_multi_rgb': pooled_rgb_features,
            'alpha_input_multi': alpha_feat_batch
        }
        predictions = multimodal_model(model_inputs_for_prediction)

        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, target_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    heatmap = target_layer_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    heatmap = heatmap.numpy()

    return heatmap

# --- 3. Function to Overlay Heatmap on Original Image ---
def save_and_display_gradcam(image_path, heatmap, alpha=0.4, title=""):
    """
    Loads the original image, overlays the heatmap, and displays the result.
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image from {image_path}. Skipping display.")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, IMAGE_SIZE)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_scaled = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)

    superimposed_img = cv2.addWeighted(img.astype(np.float32), 1 - alpha, heatmap_colored.astype(np.float32), alpha, 0)
    superimposed_img = np.uint8(superimposed_img)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(img)
    axes[0].set_title(f"Original Image\n{title}")
    axes[0].axis('off')

    axes[1].imshow(superimposed_img)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()


# --- 4. Function to Superimpose Heatmap and Return Image Array ---
def superimpose_gradcam_on_image(original_image_path, heatmap, alpha=0.4):
    """
    Loads the original image (as 3-channel RGB for display), overlays the heatmap,
    and returns the superimposed image as a NumPy array (RGB, 0-255 uint8).
    """
    # Load the original image (unmasked, for display background)
    img = tf.io.read_file(original_image_path)
    img = tf.image.decode_image(img, channels=3, expand_animations=False) # Decode as 3 channels for display
    img = tf.image.resize(img, IMAGE_SIZE)
    img_np = np.uint8(img.numpy() * 255) # Convert to 0-255 RGB NumPy array

    # Resize heatmap to original image size
    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    heatmap_scaled = np.uint8(255 * heatmap_resized) # Scale to 0-255

    # Apply a colormap to the heatmap (returns BGR, so convert to RGB)
    heatmap_colored = cv2.applyColorMap(heatmap_scaled, cv2.COLORMAP_JET)
    heatmap_colored_rgb = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Superimpose the heatmap on the original image
    superimposed_img = cv2.addWeighted(img_np.astype(np.float32), 1 - alpha, heatmap_colored_rgb.astype(np.float32), alpha, 0)
    superimposed_img = np.uint8(superimposed_img)
    
    return superimposed_img # Return the image array (RGB, uint8)

