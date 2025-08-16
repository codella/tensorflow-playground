"""
Keras Sequential Model Examples

This module demonstrates the Sequential API for building neural networks.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_simple_dense_model():
    """
    Create a simple dense (fully connected) neural network using Sequential API.
    
    This demonstrates the most basic neural network architecture where each
    layer is fully connected to the next. Perfect for learning the basics
    before moving to more complex architectures.
    """
    print("=== Simple Dense Model ===")
    
    # STEP 1: Create a Sequential model
    # Sequential models are linear stacks of layers - data flows from input to output
    model = keras.Sequential([
        
        # LAYER 1: Input layer with 64 neurons
        # - input_shape=(784,): expects flattened 28x28 images (MNIST style)
        # - 64 neurons: hidden layer size (can be tuned)
        # - activation='relu': ReLU activation function (most common)
        #   ReLU(x) = max(0, x) - prevents vanishing gradient problem
        layers.Dense(64, activation='relu', input_shape=(784,)),
        
        # LAYER 2: Hidden layer with 32 neurons
        # - Gradually reducing size is a common pattern
        # - No input_shape needed - inferred from previous layer
        layers.Dense(32, activation='relu'),
        
        # LAYER 3: Output layer with 10 neurons
        # - 10 neurons for 10 classes (digits 0-9)
        # - softmax activation: converts outputs to probabilities
        #   All outputs sum to 1.0, perfect for classification
        layers.Dense(10, activation='softmax')
        
    ], name='simple_dense_model')  # Name helps with debugging
    
    # STEP 2: Display model architecture
    # summary() shows layers, output shapes, and parameter counts
    print("Model Architecture:")
    model.summary()
    
    # Key insights from this model:
    # - Input: 784 features (28x28 flattened image)
    # - Hidden layers: 64 → 32 neurons (feature extraction)
    # - Output: 10 probabilities (one per class)
    # - Total parameters: (784×64 + 64) + (64×32 + 32) + (32×10 + 10)
    
    return model


def create_cnn_model():
    """
    Create a Convolutional Neural Network using Sequential API.
    
    CNNs are the gold standard for image processing. They use convolution
    operations to detect local features like edges, shapes, and patterns,
    making them much more effective than dense networks for images.
    """
    print("\n=== CNN Model ===")
    
    # STEP 1: Build CNN architecture
    # CNNs typically follow the pattern: Conv → Pool → Conv → Pool → Dense
    model = keras.Sequential([
        
        # CONVOLUTIONAL BLOCK 1: Feature detection
        # Conv2D layer detects basic features (edges, corners)
        layers.Conv2D(
            32,                    # 32 filters (feature detectors)
            (3, 3),               # 3x3 kernel size (filter size)
            activation='relu',     # ReLU activation
            input_shape=(28, 28, 1)  # 28x28 grayscale images
        ),
        # MaxPooling reduces spatial dimensions and adds translation invariance
        # (2,2) pool reduces 28x28 → 14x14, keeping most important features
        layers.MaxPooling2D((2, 2)),
        
        # CONVOLUTIONAL BLOCK 2: More complex features
        # Second conv layer can detect combinations of basic features
        layers.Conv2D(64, (3, 3), activation='relu'),  # More filters for complexity
        layers.MaxPooling2D((2, 2)),  # Further dimensionality reduction
        
        # CONVOLUTIONAL BLOCK 3: High-level features
        # Third conv layer detects even more complex patterns
        layers.Conv2D(64, (3, 3), activation='relu'),
        
        # TRANSITION: From spatial to dense layers
        # Flatten converts 2D feature maps to 1D vector for dense layers
        layers.Flatten(),
        
        # CLASSIFICATION HEAD: Final decision making
        # Dense layer combines all detected features for classification
        layers.Dense(64, activation='relu'),
        
        # REGULARIZATION: Prevent overfitting
        # Dropout randomly sets 50% of inputs to 0 during training
        # This prevents the model from memorizing training data
        layers.Dropout(0.5),
        
        # OUTPUT: Final classification
        # 10 neurons for 10 digit classes, softmax for probabilities
        layers.Dense(10, activation='softmax')
        
    ], name='cnn_model')
    
    # STEP 2: Display architecture
    print("CNN Architecture (notice how spatial dimensions change):")
    model.summary()
    
    # Key CNN concepts demonstrated:
    # 1. Convolution: Local feature detection with learnable filters
    # 2. Pooling: Dimensionality reduction and translation invariance
    # 3. Hierarchical features: Basic → Complex features in deeper layers
    # 4. Parameter sharing: Same filter applied across entire image
    # 5. Regularization: Dropout prevents overfitting
    
    return model


def create_rnn_model():
    """Create a Recurrent Neural Network using Sequential API."""
    print("\n=== RNN Model ===")
    
    model = keras.Sequential([
        layers.Embedding(10000, 128),
        layers.LSTM(64, dropout=0.2, recurrent_dropout=0.2),
        layers.Dense(1, activation='sigmoid')
    ], name='rnn_model')
    
    model.summary()
    return model


def create_lstm_sequence_model():
    """Create an LSTM model for sequence-to-sequence tasks."""
    print("\n=== LSTM Sequence Model ===")
    
    model = keras.Sequential([
        layers.LSTM(50, return_sequences=True, input_shape=(10, 1)),
        layers.LSTM(50, return_sequences=True),
        layers.LSTM(50),
        layers.Dense(25),
        layers.Dense(1)
    ], name='lstm_sequence_model')
    
    model.summary()
    return model


def create_autoencoder_model():
    """Create an autoencoder using Sequential API."""
    print("\n=== Autoencoder Model ===")
    
    # Encoder
    encoder = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu')
    ], name='encoder')
    
    # Decoder
    decoder = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=(32,)),
        layers.Dense(128, activation='relu'),
        layers.Dense(784, activation='sigmoid')
    ], name='decoder')
    
    # Full autoencoder
    autoencoder = keras.Sequential([
        encoder,
        decoder
    ], name='autoencoder')
    
    print("Encoder:")
    encoder.summary()
    print("\nDecoder:")
    decoder.summary()
    print("\nFull Autoencoder:")
    autoencoder.summary()
    
    return encoder, decoder, autoencoder


def demonstrate_layer_by_layer_building():
    """Demonstrate building a model layer by layer."""
    print("\n=== Layer-by-Layer Building ===")
    
    model = keras.Sequential(name='layer_by_layer_model')
    
    # Add layers one by one
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(10, activation='softmax'))
    
    # Build the model by specifying input shape
    model.build((None, 784))
    model.summary()
    
    return model


def demonstrate_model_compilation():
    """Demonstrate different ways to compile models."""
    print("\n=== Model Compilation ===")
    
    model = create_simple_dense_model()
    
    # Basic compilation
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Basic compilation:")
    print(f"Optimizer: {model.optimizer.__class__.__name__}")
    print(f"Loss: {model.loss}")
    print(f"Metrics: {model.metrics_names}")
    
    # Advanced compilation with custom parameters
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss=keras.losses.CategoricalCrossentropy(from_logits=False),
        metrics=[
            keras.metrics.CategoricalAccuracy(),
            keras.metrics.TopKCategoricalAccuracy(k=3)
        ]
    )
    print("\nAdvanced compilation:")
    print(f"Learning rate: {model.optimizer.learning_rate.numpy()}")
    print(f"Metrics: {[m.name for m in model.metrics]}")


def demonstrate_model_evaluation():
    """Demonstrate model evaluation with dummy data."""
    print("\n=== Model Evaluation ===")
    
    model = create_simple_dense_model()
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Generate dummy data
    x_test = np.random.random((100, 784))
    y_test = keras.utils.to_categorical(np.random.randint(10, size=(100,)), 10)
    
    # Evaluate model
    print("Evaluating model on dummy data...")
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Make predictions
    predictions = model.predict(x_test[:5], verbose=0)
    print(f"\nPrediction shape: {predictions.shape}")
    print(f"First prediction: {predictions[0]}")
    print(f"Predicted class: {np.argmax(predictions[0])}")


def demonstrate_model_with_regularization():
    """Demonstrate a model with various regularization techniques."""
    print("\n=== Model with Regularization ===")
    
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(784,),
                    kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(64, activation='relu',
                    kernel_regularizer=keras.regularizers.l1_l2(l1=0.001, l2=0.001)),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(10, activation='softmax')
    ], name='regularized_model')
    
    model.summary()
    return model


def demonstrate_transfer_learning_base():
    """Demonstrate creating a base for transfer learning."""
    print("\n=== Transfer Learning Base ===")
    
    # Create a base model (feature extractor)
    base_model = keras.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.GlobalAveragePooling2D()
    ], name='feature_extractor')
    
    # Freeze the base model
    base_model.trainable = False
    
    # Add classification head
    model = keras.Sequential([
        base_model,
        layers.Dropout(0.2),
        layers.Dense(10, activation='softmax')
    ], name='transfer_learning_model')
    
    print("Feature Extractor:")
    base_model.summary()
    print(f"\nBase model trainable: {base_model.trainable}")
    print(f"Number of trainable weights: {len(base_model.trainable_weights)}")
    
    print("\nFull Model:")
    model.summary()
    
    return base_model, model


def demonstrate_model_saving_and_loading():
    """Demonstrate saving and loading Sequential models."""
    print("\n=== Model Saving and Loading ===")
    
    # Create and train a simple model
    model = create_simple_dense_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Generate dummy training data
    x_train = np.random.random((100, 784))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100,)), 10)
    
    print("Training model...")
    model.fit(x_train, y_train, epochs=2, verbose=0)
    
    # Save the entire model
    model.save('temp_model.h5')
    print("Model saved as 'temp_model.h5'")
    
    # Load the model
    loaded_model = keras.models.load_model('temp_model.h5')
    print("Model loaded successfully")
    
    # Verify they produce the same outputs
    x_test = np.random.random((5, 784))
    original_pred = model.predict(x_test, verbose=0)
    loaded_pred = loaded_model.predict(x_test, verbose=0)
    
    print(f"Predictions match: {np.allclose(original_pred, loaded_pred)}")
    
    # Clean up
    import os
    if os.path.exists('temp_model.h5'):
        os.remove('temp_model.h5')


def run_all_demonstrations():
    """Run all Sequential model demonstrations."""
    create_simple_dense_model()
    create_cnn_model()
    create_rnn_model()
    create_lstm_sequence_model()
    create_autoencoder_model()
    demonstrate_layer_by_layer_building()
    demonstrate_model_compilation()
    demonstrate_model_evaluation()
    demonstrate_model_with_regularization()
    demonstrate_transfer_learning_base()
    demonstrate_model_saving_and_loading()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print()
    
    run_all_demonstrations()