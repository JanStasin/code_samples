# Multimodal Text + Image Processing Architecture

## Data Flow Overview

```
Raw Product Data
       |
       v
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA LAYER                        │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Text Tokens   │    │   Image Paths   │                │
│  │ [15,847,23,...] │    │ "img/prod1.jpg" │                │
│  └─────────────────┘    └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘
       |                           |
       v                           v
```
```

## Data Generator Flow

```
┌─────────────────────────────────────────────────────────────┐
│               MULTIMODAL DATA GENERATOR                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  DataFrame Input:                                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ ProductID | comb_tokens_fr  | imagepath | label    │    │
│  │    001    | [15,847,23,...] | img/1.jpg |   5      │    │
│  │    002    | [22,156,88,...] | img/2.jpg |   12     │    │
│  │    ...    |      ...        |    ...    |  ...     │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  For each batch (e.g., 32 products):                       │
│                                                             │
│  Step 1: Select batch indices [64, 65, 66, ..., 95]        │
│  Step 2: Create empty arrays                                │
│         - text_batch: (32, 60)                             │
│         - image_batch: (32, 224, 224, 3)                   │
│         - label_batch: (32, 27)                            │
│                                                             │
│  Step 3: For each product in batch:                        │
│         - Fill text_batch[i] with token sequence           │
│         - Load & preprocess image into image_batch[i]      │
│         - Fill label_batch[i] with one-hot label           │
│                                                             │
│  Step 4: Return ((text_batch, image_batch), label_batch)   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```
## Detailed Architecture Schematic

### TEXT PROCESSING BRANCH
```
┌─────────────────────────────────────────────────────────────┐
│                     TEXT MODALITY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Token Sequences                                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           [15, 847, 23, 156, 0, 0, ...]            │    │
│  │              Shape: (batch, 60)                    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              EMBEDDING LAYER                       │    │
│  │         Input: (batch, 60)                         │    │
│  │         Output: (batch, 60, 128)                   │    │
│  │         Vocab Size: 15,001                         │    │
│  │         Embedding Dim: 128                         │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            PARALLEL CONV1D BRANCHES                 │    │
│  │                                                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │ Conv1D  │  │ Conv1D  │  │ Conv1D  │              │    │
│  │  │ k=2     │  │ k=3     │  │ k=4     │              │    │
│  │  │ f=64    │  │ f=64    │  │ f=64    │              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  │       |           |           |                     │    │
│  │       v           v           v                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │BatchNorm│  │BatchNorm│  │BatchNorm│              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  │       |           |           |                     │    │
│  │       v           v           v                     │    │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐              │    │
│  │  │GlobalMax│  │GlobalMax│  │GlobalMax│              │    │
│  │  │Pooling1D│  │Pooling1D│  │Pooling1D│              │    │
│  │  └─────────┘  └─────────┘  └─────────┘              │    │
│  │       |           |           |                     │    │
│  │       └───────────┼───────────┘                     │    │
│  │                   v                                 │    │
│  │  ┌─────────────────────────────────────────────────┐     │
│  │  │            CONCATENATE                          │     │
│  │  │         Output: (batch, 192)                    │     │
│  │  │         [64+64+64 features]                     │     │
│  │  └─────────────────────────────────────────────────┘     │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              TEXT DENSE LAYERS                      │    │
│  │                                                     │    │
│  │  Dense(128) + BatchNorm + Dropout(0.3)              │    │
│  │                                                     │    │
│  │         Output: (batch, 128)                        │    │
│  │         = TEXT FEATURES                             │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### IMAGE PROCESSING BRANCH
```
┌─────────────────────────────────────────────────────────────┐
│                     IMAGE MODALITY                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Input: Raw Images                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Load from disk                         │    │
│  │          "img/product_001.jpg"                      │    │
│  │              Shape: (224, 224, 3)                   │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           PREPROCESSING                             │    │
│  │                                                     │    │
│  │  • Resize to 224x224                                │    │
│  │  • MobileNetV2 preprocess_input()                   │    │
│  │  • Normalize to [-1, 1] range                       │    │
│  │  • Convert to float32                               │    │
│  │                                                     │    │
│  │         Output: (batch, 224, 224, 3)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │            MOBILENETV2 / EffNetB0 BACKBONE          │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │     Pretrained on ImageNet                  │    │    │
│  │  │     Frozen: True (initially)                │    │    │
│  │  │     Input: (batch, 224, 224, 3)             │    │    │
│  │  │     Output: (batch, 7, 7, 1280)             │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                          |                          │    │
│  │                          v                          │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │       GLOBAL AVERAGE POOLING 2D             │    │    │
│  │  │       Input: (batch, 7, 7, 1280)            │    │    │
│  │  │       Output: (batch, 1280)                 │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │             IMAGE DENSE LAYERS                      │    │
│  │                                                     │    │
│  │  Dense(256) + BatchNorm + Dropout(0.3)              │    │
│  │                                                     │    │
│  │         Output: (batch, 256)                        │    │ 
│  │         = IMAGE FEATURES                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

### MULTIMODAL FUSION
```
┌─────────────────────────────────────────────────────────────┐
│                   MULTIMODAL FUSION                        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Text Features        Image Features                        │
│  ┌─────────────┐      ┌─────────────┐                       │
│  │ (batch,128) │      │ (batch,256) │                       │
│  └─────────────┘      └─────────────┘                       │
│         |                     |                             │
│         └─────────┬───────────┘                             │
│                   v                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              CONCATENATE                            │    │
│  │         Output: (batch, 384)                        │    │
│  │         [128 + 256 features]                        │    │
│  └─────────────────────────────────────────────────────┘    │
│                          |                                  │
│                          v                                  │
│  ┌─────────────────────────────────────────────────────┐    │
│  │           SHARED CLASSIFICATION LAYERS              │    │
│  │                                                     │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Dense( N ) + BatchNorm + Dropout(0.4)       │    │    │
│  │  │ Input: (batch, 384)                         │    │    │
│  │  │ Output: (batch, 256)                        │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                          |                          │    │
│  │                          v                          │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Dense( N ) + BatchNorm + Dropout(0.3)       │    │    │
│  │  │ Input: (batch, 256)                         │    │    │
│  │  │ Output: (batch, 128)                        │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  │                          |                          │    │
│  │                          v                          │    │
│  │  ┌─────────────────────────────────────────────┐    │    │
│  │  │ Dense(27) + Softmax                         │    │    │
│  │  │ Input: (batch, 128)                         │    │    │
│  │  │ Output: (batch, 27)                         │    │    │
│  │  │ = FINAL PREDICTIONS                         │    │    │
│  │  └─────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘


## Training Process

```
┌─────────────────────────────────────────────────────────────┐
│                    TRAINING FLOW                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Phase 1: Frozen MobileNetV2 / EffNetB0                     │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • MobileNetV2: Frozen (weights don't change)        │    │
│  │ • Text CNN: Trainable                               │    │
│  │ • Fusion layers: Trainable                          │    │
│  │ • Learning rate: 0.001                              │    │
│  │ • Focus: Learn text features + fusion strategy      │    │
│  └─────────────────────────────────────────────────────┘    │
│                                                             │
│  Phase 2: Fine-tuning (Optional)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • MobileNetV2: Unfrozen                             │    │
│  │ • All layers: Trainable                             │    │
│  │ • Learning rate: 0.0001 (lower)                     │    │
│  │ • Focus: Fine-tune entire model end-to-end          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## Key Benefits

### Individual Modalities:
- **Text CNN**: Captures semantic meaning from product descriptions
- **Pretrained Image CNNS**:
    **MobileNetV2**: Extracts visual features from product images
    **EfficientNetB0** : Extracts visual features from product images

### Multimodal Fusion:
- **Complementary Information**: Text describes what image might not show
- **Robust Predictions**: Works even if one modality fails
- **Richer Representation**: 384-dimensional fused features vs 128 or 256 alone
- **Better Accuracy**: Typically 5-15% improvement over single modality

### Architecture Advantages:
- **Scalable**: Can add more modalities (price, reviews, etc.)
- **Flexible**: Can train modalities separately or jointly
- **Efficient**: MobileNetV2 provides good speed/accuracy trade-off
- **Practical**: Handles missing data gracefully