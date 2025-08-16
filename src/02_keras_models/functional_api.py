"""
Keras Functional API Examples

This module demonstrates the Functional API for building complex neural networks.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


def create_simple_functional_model():
    """Create a simple model using the Functional API."""
    print("=== Simple Functional Model ===")
    
    # Define input
    inputs = keras.Input(shape=(784,), name='input_layer')
    
    # Define layers
    x = layers.Dense(64, activation='relu', name='hidden_1')(inputs)
    x = layers.Dense(32, activation='relu', name='hidden_2')(x)
    outputs = layers.Dense(10, activation='softmax', name='output_layer')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, outputs=outputs, name='simple_functional_model')
    
    model.summary()
    keras.utils.plot_model(model, show_shapes=True, show_layer_names=True, 
                          to_file='simple_functional_model.png')
    print("Model diagram saved as 'simple_functional_model.png'")
    
    return model


def create_multi_input_model():
    """Create a model with multiple inputs."""
    print("\n=== Multi-Input Model ===")
    
    # Define multiple inputs
    text_input = keras.Input(shape=(100,), name='text')
    image_input = keras.Input(shape=(28, 28, 1), name='image')
    
    # Process text input
    text_features = layers.Dense(64, activation='relu')(text_input)
    text_features = layers.Dropout(0.3)(text_features)
    
    # Process image input
    image_features = layers.Conv2D(32, 3, activation='relu')(image_input)
    image_features = layers.GlobalMaxPooling2D()(image_features)
    image_features = layers.Dense(64, activation='relu')(image_features)
    image_features = layers.Dropout(0.3)(image_features)
    
    # Concatenate features
    concatenated = layers.Concatenate()([text_features, image_features])
    
    # Final classification layers
    x = layers.Dense(64, activation='relu')(concatenated)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    # Create model
    model = keras.Model(inputs=[text_input, image_input], outputs=outputs, 
                       name='multi_input_model')
    
    model.summary()
    
    # Test with dummy data
    text_data = np.random.random((32, 100))
    image_data = np.random.random((32, 28, 28, 1))
    
    predictions = model([text_data, image_data])
    print(f"Prediction shape: {predictions.shape}")
    
    return model


def create_multi_output_model():
    """Create a model with multiple outputs."""
    print("\n=== Multi-Output Model ===")
    
    # Shared input
    inputs = keras.Input(shape=(784,), name='input')
    
    # Shared layers
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    
    # Multiple output branches
    classification_output = layers.Dense(10, activation='softmax', 
                                       name='classification')(x)
    regression_output = layers.Dense(1, activation='linear', 
                                   name='regression')(x)
    
    # Create model
    model = keras.Model(inputs=inputs, 
                       outputs=[classification_output, regression_output],
                       name='multi_output_model')
    
    model.summary()
    
    # Compile with different losses for each output
    model.compile(
        optimizer='adam',
        loss={
            'classification': 'categorical_crossentropy',
            'regression': 'mse'
        },
        metrics={
            'classification': ['accuracy'],
            'regression': ['mae']
        },
        loss_weights={
            'classification': 1.0,
            'regression': 0.5
        }
    )
    
    return model


def create_residual_block_model():
    """Create a model with residual connections."""
    print("\n=== Residual Block Model ===")
    
    def residual_block(x, filters, kernel_size=3):
        """Create a residual block."""
        # Save input for skip connection
        shortcut = x
        
        # Main path
        x = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, kernel_size, padding='same')(x)
        x = layers.BatchNormalization()(x)
        
        # Adjust shortcut if needed
        if shortcut.shape[-1] != filters:
            shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
        
        # Add skip connection
        x = layers.Add()([x, shortcut])
        x = layers.Activation('relu')(x)
        
        return x
    
    # Build model with residual blocks
    inputs = keras.Input(shape=(32, 32, 3))
    
    x = layers.Conv2D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Add residual blocks
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    
    # Final layers
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(10, activation='softmax')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='residual_model')
    model.summary()
    
    return model


def create_attention_model():
    """Create a simple attention mechanism model."""
    print("\n=== Attention Model ===")
    
    # Input
    inputs = keras.Input(shape=(100, 64), name='sequence_input')  # (seq_len, features)
    
    # LSTM encoder
    lstm_out = layers.LSTM(128, return_sequences=True)(inputs)
    
    # Attention mechanism
    attention_weights = layers.Dense(1, activation='tanh')(lstm_out)
    attention_weights = layers.Flatten()(attention_weights)
    attention_weights = layers.Activation('softmax')(attention_weights)
    attention_weights = layers.RepeatVector(128)(attention_weights)
    attention_weights = layers.Permute([2, 1])(attention_weights)
    
    # Apply attention
    attention_output = layers.Multiply()([lstm_out, attention_weights])
    attention_output = layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(attention_output)
    
    # Final classification
    outputs = layers.Dense(10, activation='softmax')(attention_output)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name='attention_model')
    model.summary()
    
    return model


def create_autoencoder_functional():
    """Create an autoencoder using Functional API."""
    print("\n=== Functional Autoencoder ===")
    
    # Encoder
    encoder_input = keras.Input(shape=(784,), name='encoder_input')
    encoded = layers.Dense(128, activation='relu')(encoder_input)
    encoded = layers.Dense(64, activation='relu')(encoded)
    encoded = layers.Dense(32, activation='relu', name='bottleneck')(encoded)
    
    encoder = keras.Model(encoder_input, encoded, name='encoder')
    
    # Decoder
    decoder_input = keras.Input(shape=(32,), name='decoder_input')
    decoded = layers.Dense(64, activation='relu')(decoder_input)
    decoded = layers.Dense(128, activation='relu')(decoded)
    decoded = layers.Dense(784, activation='sigmoid')(decoded)
    
    decoder = keras.Model(decoder_input, decoded, name='decoder')
    
    # Full autoencoder
    autoencoder_input = keras.Input(shape=(784,), name='autoencoder_input')
    encoded_repr = encoder(autoencoder_input)
    decoded_output = decoder(encoded_repr)
    autoencoder = keras.Model(autoencoder_input, decoded_output, name='autoencoder')
    
    print("Encoder:")
    encoder.summary()
    print("\nDecoder:")
    decoder.summary()
    print("\nAutoencoder:")
    autoencoder.summary()
    
    return encoder, decoder, autoencoder


def create_siamese_network():
    """Create a Siamese network for similarity learning."""
    print("\n=== Siamese Network ===")
    
    def create_base_network(input_shape):
        """Create the base network for feature extraction."""
        input_layer = keras.Input(shape=input_shape)
        x = layers.Dense(128, activation='relu')(input_layer)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.1)(x)
        x = layers.Dense(128, activation='relu')(x)
        return keras.Model(input_layer, x)
    
    # Create base network
    base_network = create_base_network((784,))
    
    # Define inputs for the siamese network
    input_a = keras.Input(shape=(784,), name='input_a')
    input_b = keras.Input(shape=(784,), name='input_b')
    
    # Process both inputs through the same base network
    processed_a = base_network(input_a)
    processed_b = base_network(input_b)
    
    # Compute distance between the two processed inputs
    distance = layers.Lambda(lambda x: tf.abs(x[0] - x[1]))([processed_a, processed_b])
    
    # Final prediction
    outputs = layers.Dense(1, activation='sigmoid')(distance)
    
    siamese_model = keras.Model([input_a, input_b], outputs, name='siamese_network')
    
    print("Base network:")
    base_network.summary()
    print("\nSiamese network:")
    siamese_model.summary()
    
    return base_network, siamese_model


def create_variational_autoencoder():
    """Create a Variational Autoencoder using Functional API."""
    print("\n=== Variational Autoencoder ===")
    
    latent_dim = 32
    
    # Encoder
    encoder_inputs = keras.Input(shape=(784,), name='encoder_input')
    x = layers.Dense(256, activation='relu')(encoder_inputs)
    x = layers.Dense(128, activation='relu')(x)
    
    # Latent space parameters
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    
    # Sampling function
    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
    
    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    
    # Encoder model
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    
    # Decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(128, activation='relu')(latent_inputs)
    x = layers.Dense(256, activation='relu')(x)
    decoder_outputs = layers.Dense(784, activation='sigmoid')(x)
    
    decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
    
    # VAE model
    outputs = decoder(encoder(encoder_inputs)[2])
    vae = keras.Model(encoder_inputs, outputs, name='vae')
    
    print("Encoder:")
    encoder.summary()
    print("\nDecoder:")
    decoder.summary()
    print("\nVAE:")
    vae.summary()
    
    return encoder, decoder, vae


def demonstrate_model_manipulation():
    """Demonstrate various model manipulation techniques."""
    print("\n=== Model Manipulation ===")
    
    # Create a model
    inputs = keras.Input(shape=(784,))
    x = layers.Dense(128, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(x)
    x = layers.Dense(32, activation='relu', name='dense_3')(x)
    outputs = layers.Dense(10, activation='softmax', name='dense_4')(x)
    
    model = keras.Model(inputs, outputs, name='manipulation_model')
    
    # Get specific layers
    dense_2 = model.get_layer('dense_2')
    print(f"Dense_2 layer: {dense_2}")
    print(f"Dense_2 output shape: {dense_2.output_shape}")
    
    # Create a new model using intermediate layer as output
    feature_extractor = keras.Model(inputs=model.input, 
                                  outputs=model.get_layer('dense_3').output,
                                  name='feature_extractor')
    print("\nFeature extractor model:")
    feature_extractor.summary()
    
    # Freeze certain layers
    for layer in model.layers[:-1]:  # Freeze all except last layer
        layer.trainable = False
    
    print(f"\nTrainable weights after freezing: {len(model.trainable_weights)}")
    
    # Get model configuration
    config = model.get_config()
    print(f"Model config keys: {list(config.keys())}")


def run_all_demonstrations():
    """Run all Functional API demonstrations."""
    create_simple_functional_model()
    create_multi_input_model()
    create_multi_output_model()
    create_residual_block_model()
    create_attention_model()
    create_autoencoder_functional()
    create_siamese_network()
    create_variational_autoencoder()
    demonstrate_model_manipulation()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print()
    
    run_all_demonstrations()