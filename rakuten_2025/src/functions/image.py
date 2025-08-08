
import os
import base64

# --- Function to convert image to Base64 ---
def get_image_base64(image_path):
    """Converts an image file to a Base64 string for embedding in HTML."""
    if not os.path.exists(image_path):
        return None, None # Image not found

    try:
        # Determine image format from extension for the data URI
        file_extension = os.path.splitext(image_path)[1].lower()
        if file_extension == '.jpg' or file_extension == '.jpeg':
            mime_type = 'image/jpeg'
        elif file_extension == '.png':
            mime_type = 'image/png'
        elif file_extension == '.gif':
            mime_type = 'image/gif'
        else:
            # Fallback or raise error for unsupported formats
            print(f"Warning: Unsupported image format for {image_path}. Attempting as JPEG.")
            mime_type = 'image/jpeg' # Default to jpeg

        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        return encoded_string, mime_type
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None, None