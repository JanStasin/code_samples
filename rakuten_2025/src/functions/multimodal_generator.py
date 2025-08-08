import numpy as np
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

class MultimodalDataGenerator(Sequence):
    """Custom data generator that handles both text sequences and images simultaneously
    Takes a batch of product IDs → Returns text sequences + images + labels for those products
    """
    
    def __init__(self, df, text_sequences, batch_size, img_size, use_mobilenet_prep=True, 
                 augment_images=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.text_sequences = text_sequences
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_mobilenet_prep = use_mobilenet_prep
        self.augment_images = augment_images
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.n_classes = len(self.df['prdtypecode'].unique())
        
        # Add compatibility attributes for training_report.py
        self.classes = self.df['prdtypecode'].astype('category').cat.codes.values
        unique_classes = sorted(self.df['prdtypecode'].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.target_size = (self.img_size, self.img_size)
        
        self.on_epoch_end()
    
    def __len__(self):
        """Include all samples, even if last batch is smaller"""
        return (len(self.df) + self.batch_size - 1) // self.batch_size
    
    def __getitem__(self, index):
        # Handle last batch being potentially smaller
        start_idx = index * self.batch_size
        end_idx = min((index + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start_idx:end_idx]
        actual_batch_size = len(batch_indices)
        
        # Initialize batch arrays with actual batch size
        text_batch = np.zeros((actual_batch_size, self.text_sequences.shape[1]), dtype=np.float32)
        image_batch = np.zeros((actual_batch_size, self.img_size, self.img_size, 3), dtype=np.float32)
        label_batch = np.zeros((actual_batch_size, self.n_classes), dtype=np.float32)
        
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
                if self.use_mobilenet_prep:
                    img_array = preprocess_input(img_array)
                else:
                    img_array = img_array / 255.0
                    
                image_batch[i] = img_array.astype(np.float32)
            except:
                # Handle missing/corrupt images with zeros (will be preprocessed accordingly)
                if self.use_mobilenet_prep:
                    zero_img = np.zeros((self.img_size, self.img_size, 3))
                    image_batch[i] = preprocess_input(zero_img).astype(np.float32)
                else:
                    image_batch[i] = np.zeros((self.img_size, self.img_size, 3), dtype=np.float32)
            
            # Load label (one-hot encoded)
            label = self.df.iloc[idx]['prdtypecode_categorical']
            label_batch[i] = np.array(label, dtype=np.float32)
        
        return (text_batch, image_batch), label_batch
    
    def on_epoch_end(self):
        """Shuffle indices after each epoch if shuffle=True"""
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def reset(self):
        """Reset generator for training_report compatibility"""
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)


class MultimodalDataGeneratorAug(Sequence):
    def __init__(self, df, text_sequences, batch_size=32, img_size=224, 
                 use_mobilenet_prep=True, augment_images=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.text_sequences = text_sequences
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_mobilenet_prep = use_mobilenet_prep
        self.augment_images = augment_images
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.n_classes = len(self.df['prdtypecode'].unique())
        
        # Add compatibility attributes for training_report.py
        self.classes = self.df['prdtypecode'].astype('category').cat.codes.values
        unique_classes = sorted(self.df['prdtypecode'].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.target_size = (self.img_size, self.img_size)
        
        self.on_epoch_end()

        if self.augment_images:
            self.augmentor = ImageDataGenerator(
                rotation_range=15,              # Rotate up to 15 degrees
                zoom_range=0.1,                 # Zoom up to 10%
                horizontal_flip=True,           # Flip horizontally
                brightness_range=[0.95, 1.05],  # Subtle brightness change
                fill_mode='nearest'
            )
        else:
            self.augmentor = None
            
        self.indices = np.arange(len(self.df))
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_size_actual = len(batch_indices)
        text_batch = np.zeros((batch_size_actual, self.text_sequences.shape[1]))
        img_batch = np.zeros((batch_size_actual, self.img_size, self.img_size, 3))
        label_batch = np.zeros((batch_size_actual, len(self.df['prdtypecode_categorical'].iloc[0])))
        
        for i, data_idx in enumerate(batch_indices):
            text_batch[i] = self.text_sequences[data_idx]
            
            img_path = self.df.iloc[data_idx]['imagepath']
            
            try:
                img = load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img)
                
                if self.augment_images and self.augmentor is not None:
                    img_array = self.augmentor.random_transform(img_array)
                
                if self.use_mobilenet_prep:
                    img_array = preprocess_input(img_array)
                else:
                    img_array = img_array / 255.0
                    
                img_batch[i] = img_array
                
            except Exception as e:
                print(f'Error loading image {img_path}: {e}')
                img_batch[i] = np.zeros((self.img_size, self.img_size, 3))
            
            label_batch[i] = self.df.iloc[data_idx]['prdtypecode_categorical']
        
        return (text_batch, img_batch), label_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        """Reset generator for training_report compatibility"""
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)



class ImageDataGeneratorAug(Sequence):
    """Image-only data generator with augmentation support"""
    
    def __init__(self, df, batch_size=32, img_size=224, 
                 use_mobilenet_prep=True, augment_images=False, shuffle=True, **kwargs):
        super().__init__(**kwargs)
        self.df = df.reset_index(drop=True)
        self.batch_size = batch_size
        self.img_size = img_size
        self.use_mobilenet_prep = use_mobilenet_prep
        self.augment_images = augment_images
        self.shuffle = shuffle
        self.indices = np.arange(len(self.df))
        self.n_classes = len(self.df['prdtypecode'].unique())
        
        # Add compatibility attributes for training_report.py
        self.classes = self.df['prdtypecode'].astype('category').cat.codes.values
        unique_classes = sorted(self.df['prdtypecode'].unique())
        self.class_indices = {class_name: i for i, class_name in enumerate(unique_classes)}
        self.target_size = (self.img_size, self.img_size)
        
        # Setup image augmentation
        if self.augment_images:
            self.augmentor = ImageDataGenerator(
                rotation_range=15,              # Rotate up to 15 degrees
                zoom_range=0.1,                 # Zoom up to 10%
                horizontal_flip=True,           # Flip horizontally
                brightness_range=[0.95, 1.05],  # Subtle brightness change
                fill_mode='nearest'
            )
        else:
            self.augmentor = None
            
        self.on_epoch_end()
    
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.df))
        batch_indices = self.indices[start_idx:end_idx]
        
        batch_size_actual = len(batch_indices)
        img_batch = np.zeros((batch_size_actual, self.img_size, self.img_size, 3))
        label_batch = np.zeros((batch_size_actual, len(self.df['prdtypecode_categorical'].iloc[0])))
        
        for i, data_idx in enumerate(batch_indices):
            img_path = self.df.iloc[data_idx]['imagepath']
            
            try:
                img = load_img(img_path, target_size=(self.img_size, self.img_size))
                img_array = img_to_array(img)
                
                # Apply augmentation if enabled
                if self.augment_images and self.augmentor is not None:
                    img_array = self.augmentor.random_transform(img_array)
                
                # Apply preprocessing
                if self.use_mobilenet_prep:
                    img_array = preprocess_input(img_array)
                else:
                    img_array = img_array / 255.0
                    
                img_batch[i] = img_array
                
            except Exception as e:
                print(f'Error loading image {img_path}: {e}')
                # Handle missing/corrupt images with zeros
                if self.use_mobilenet_prep:
                    zero_img = np.zeros((self.img_size, self.img_size, 3))
                    img_batch[i] = preprocess_input(zero_img)
                else:
                    img_batch[i] = np.zeros((self.img_size, self.img_size, 3))
            
            label_batch[i] = self.df.iloc[data_idx]['prdtypecode_categorical']
        
        return img_batch, label_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def reset(self):
        """Reset generator for training_report compatibility"""
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)