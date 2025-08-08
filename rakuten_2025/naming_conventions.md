naming_conventions.md


DATA:
data/
X_train_update.csv 
    - raw input train data 
    - columns: 	['Unnamed: 0', 'designation', 'description',	'productid',	'imageid']

X_test_update.csv
    - raw input test data 
    - columns: 	['Unnamed: 0', 'designation', 'description',	'productid',	'imageid']

Y_train_CVw08PX.csv
    - raw input train labels
    - columns: [Unnamed: 0	prdtypecode]

PROCESSED:
processed_data/ 

****** you need to unzip this file
X_train_with_labels.csv
    - processed data with labels --> 'product_id'
    - additional columns for French and English category names, and a combined text field for model input
    - columns: ['designation', 'description', 'productid', 'imageid',
     'prdtypecode', 'cat_name_fr', 'cat_name_en', 'combined']
    - generated from: 'text_exploration.ipynb'
    
all_stopwords.npy
    - list of stopwords used in text preprocessing
    - generated from: 'text_exploration.ipynb'

flattened_images_{N}.csv
    N: 32,64,128,500
    -generate from img_exploration

RESULTS:
image_analysis_results_bgrm.csv
    - results of image analysis
    - columns: ['productid', 'imageid', 'imagepath', 'exists', 'width', 'height',
       'sharpness', 'is_sharp', 'brightness', 'is_bright', 'average_rgb',
       'color_type', 'contrast']
    - generated from: 'img_exploration.ipynb'


NOTEBOOKS:
notebooks/

text_exploration.ipynb
    - text analysis and preprocessing
    - former: 2.0-simon-data-exploration-EN.ipynb
img_exploration.ipynb
    - image analysis and processing
    - former: images_exploration
vectorization.ipynb
    - exploration of different text vectorization and embeddings strategies
text_modeling.ipynb
    - playground for building and testing text-only models
img_modeling.ipynb
    - playground for building and testing image-only models
    - to be created
text_and_image_modeling.ipynb
    - playground for building and testing text and image models
main_modeling.ipynb
    - notebook for working and high performance text and image models


HELPER_FUCNTIONS
src/functions/

text_helpers.py
    - custom functions used in text-notebooks
image_helpers.py
    - custom functions used in img-notebooks








