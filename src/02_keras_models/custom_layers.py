"""
Keras Custom Layers Examples

This module demonstrates how to create custom layers in Keras.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class LinearLayer(layers.Layer):
    """A simple linear layer (fully connected)."""
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
    
    def build(self, input_shape):
        # Create the weights
        self.w = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='random_normal',
            trainable=True
        )
        self.b = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True
        )
        super().build(input_shape)
    
    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
    
    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})
        return config


class DropoutLayer(layers.Layer):
    """A custom dropout layer."""
    
    def __init__(self, rate, **kwargs):
        super().__init__(**kwargs)
        self.rate = rate
    
    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate)
        return inputs
    
    def get_config(self):
        config = super().get_config()
        config.update({"rate": self.rate})
        return config


class MultiHeadAttention(layers.Layer):
    """A simplified multi-head attention layer."""
    
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, inputs):
        q, k, v = inputs, inputs, inputs  # Self-attention
        batch_size = tf.shape(q)[0]
        
        # Linear projections
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        
        # Split into heads
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        
        # Scaled dot-product attention
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        
        # Concatenate heads
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        
        # Final linear projection
        output = self.dense(concat_attention)
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads
        })
        return config


class PositionalEmbedding(layers.Layer):
    """Positional embedding layer for transformer models."""
    
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        
        self.token_embeddings = layers.Embedding(vocab_size, embed_dim)
        self.position_embeddings = layers.Embedding(sequence_length, embed_dim)
    
    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "embed_dim": self.embed_dim
        })
        return config


class ResidualConnection(layers.Layer):
    """Residual connection with layer normalization."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.layer_norm = layers.LayerNormalization()
        self.add = layers.Add()
    
    def call(self, inputs):
        input_tensor, output_tensor = inputs
        return self.layer_norm(self.add([input_tensor, output_tensor]))


class TransformerBlock(layers.Layer):
    """A transformer block."""
    
    def __init__(self, d_model, num_heads, dff, rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.rate = rate
        
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(dff, activation='relu'),
            layers.Dense(d_model)
        ])
        
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)
    
    def call(self, inputs, training=None):
        # Multi-head attention
        attn_output = self.mha(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed forward network
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        
        return out2
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "dff": self.dff,
            "rate": self.rate
        })
        return config


class ConvolutionalBlock(layers.Layer):
    """A convolutional block with batch normalization and activation."""
    
    def __init__(self, filters, kernel_size=3, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_name = activation
        
        self.conv = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')
        self.bn = layers.BatchNormalization()
        self.activation = layers.Activation(activation)
    
    def call(self, inputs, training=None):
        x = self.conv(inputs)
        x = self.bn(x, training=training)
        return self.activation(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides,
            "activation": self.activation_name
        })
        return config


class SeparableConvBlock(layers.Layer):
    """Depthwise separable convolution block."""
    
    def __init__(self, filters, kernel_size=3, strides=1, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        
        self.depthwise_conv = layers.DepthwiseConv2D(
            kernel_size, strides=strides, padding='same'
        )
        self.pointwise_conv = layers.Conv2D(filters, 1, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.ReLU()
    
    def call(self, inputs, training=None):
        x = self.depthwise_conv(inputs)
        x = self.bn1(x, training=training)
        x = self.activation(x)
        
        x = self.pointwise_conv(x)
        x = self.bn2(x, training=training)
        x = self.activation(x)
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size,
            "strides": self.strides
        })
        return config


class SqueezeAndExcitation(layers.Layer):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.ratio = ratio
    
    def build(self, input_shape):
        self.channels = input_shape[-1]
        self.squeeze = layers.GlobalAveragePooling2D()
        self.excite = keras.Sequential([
            layers.Dense(self.channels // self.ratio, activation='relu'),
            layers.Dense(self.channels, activation='sigmoid')
        ])
        super().build(input_shape)
    
    def call(self, inputs):
        # Squeeze
        se = self.squeeze(inputs)
        
        # Excitation
        se = self.excite(se)
        se = tf.expand_dims(se, 1)
        se = tf.expand_dims(se, 1)
        
        # Scale
        return inputs * se
    
    def get_config(self):
        config = super().get_config()
        config.update({"ratio": self.ratio})
        return config


class GatedLinearUnit(layers.Layer):
    """Gated Linear Unit (GLU) activation."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def build(self, input_shape):
        # Ensure even number of features for splitting
        assert input_shape[-1] % 2 == 0
        super().build(input_shape)
    
    def call(self, inputs):
        # Split input into two halves
        a, b = tf.split(inputs, 2, axis=-1)
        return a * tf.nn.sigmoid(b)


class SpectralNormalization(layers.Wrapper):
    """Spectral normalization wrapper for any layer."""
    
    def __init__(self, layer, **kwargs):
        super().__init__(layer, **kwargs)
    
    def build(self, input_shape):
        super().build(input_shape)
        
        if hasattr(self.layer, 'kernel'):
            self.kernel = self.layer.kernel
            self.kernel_shape = self.kernel.shape
            
            # Initialize u vector
            self.u = self.add_weight(
                name='u',
                shape=(1, self.kernel_shape[-1]),
                initializer='random_normal',
                trainable=False
            )
    
    def call(self, inputs, training=None):
        if hasattr(self, 'kernel'):
            # Power iteration method
            w_reshaped = tf.reshape(self.kernel, [-1, self.kernel_shape[-1]])
            u = self.u
            
            for _ in range(1):  # Usually 1 iteration is enough
                v = tf.nn.l2_normalize(tf.matmul(u, w_reshaped, transpose_b=True))
                u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped))
            
            # Compute spectral norm
            sigma = tf.reduce_sum(tf.matmul(u, w_reshaped, transpose_b=True) * v)
            
            # Normalize weights
            w_normalized = self.kernel / sigma
            
            # Replace the layer's kernel temporarily
            original_kernel = self.layer.kernel
            self.layer.kernel = w_normalized
            output = self.layer(inputs, training=training)
            self.layer.kernel = original_kernel
            
            # Update u
            if training:
                self.u.assign(u)
            
            return output
        else:
            return self.layer(inputs, training=training)


def demonstrate_custom_layers():
    """Demonstrate various custom layers."""
    print("=== Custom Layers Demonstration ===")
    
    # Test simple linear layer
    print("\n1. Linear Layer:")
    linear = LinearLayer(units=64)
    x = tf.random.normal((10, 32))
    output = linear(x)
    print(f"Input shape: {x.shape}, Output shape: {output.shape}")
    
    # Test custom dropout
    print("\n2. Custom Dropout Layer:")
    dropout = DropoutLayer(rate=0.5)
    output_train = dropout(x, training=True)
    output_test = dropout(x, training=False)
    print(f"Training mode: {tf.reduce_mean(output_train):.4f}")
    print(f"Test mode: {tf.reduce_mean(output_test):.4f}")
    
    # Test multi-head attention
    print("\n3. Multi-Head Attention:")
    attention = MultiHeadAttention(d_model=64, num_heads=8)
    seq_input = tf.random.normal((2, 10, 64))  # (batch, seq_len, d_model)
    attention_output = attention(seq_input)
    print(f"Input shape: {seq_input.shape}, Output shape: {attention_output.shape}")
    
    # Test positional embedding
    print("\n4. Positional Embedding:")
    pos_embed = PositionalEmbedding(sequence_length=100, vocab_size=10000, embed_dim=64)
    tokens = tf.random.uniform((2, 50), maxval=10000, dtype=tf.int32)
    embedded = pos_embed(tokens)
    print(f"Token shape: {tokens.shape}, Embedded shape: {embedded.shape}")
    
    # Test transformer block
    print("\n5. Transformer Block:")
    transformer = TransformerBlock(d_model=64, num_heads=8, dff=256)
    transformer_output = transformer(seq_input)
    print(f"Input shape: {seq_input.shape}, Output shape: {transformer_output.shape}")
    
    # Test convolutional block
    print("\n6. Convolutional Block:")
    conv_block = ConvolutionalBlock(filters=32, kernel_size=3)
    img_input = tf.random.normal((2, 28, 28, 3))
    conv_output = conv_block(img_input)
    print(f"Input shape: {img_input.shape}, Output shape: {conv_output.shape}")
    
    # Test separable conv block
    print("\n7. Separable Convolution Block:")
    sep_conv = SeparableConvBlock(filters=64, kernel_size=3)
    sep_output = sep_conv(img_input)
    print(f"Input shape: {img_input.shape}, Output shape: {sep_output.shape}")
    
    # Test squeeze and excitation
    print("\n8. Squeeze-and-Excitation:")
    se_block = SqueezeAndExcitation(ratio=16)
    se_output = se_block(conv_output)
    print(f"Input shape: {conv_output.shape}, Output shape: {se_output.shape}")
    
    # Test GLU
    print("\n9. Gated Linear Unit:")
    glu = GatedLinearUnit()
    glu_input = tf.random.normal((5, 128))  # Even number of features
    glu_output = glu(glu_input)
    print(f"Input shape: {glu_input.shape}, Output shape: {glu_output.shape}")


def create_model_with_custom_layers():
    """Create a model using custom layers."""
    print("\n=== Model with Custom Layers ===")
    
    # Create a simple CNN with custom layers
    model = keras.Sequential([
        ConvolutionalBlock(32, kernel_size=3, strides=1),
        SqueezeAndExcitation(ratio=8),
        layers.MaxPooling2D(2),
        
        SeparableConvBlock(64, kernel_size=3),
        SqueezeAndExcitation(ratio=8),
        layers.MaxPooling2D(2),
        
        ConvolutionalBlock(128, kernel_size=3),
        layers.GlobalAveragePooling2D(),
        
        LinearLayer(64),
        DropoutLayer(0.5),
        LinearLayer(10),
        layers.Activation('softmax')
    ], name='custom_cnn')
    
    # Build the model
    model.build((None, 28, 28, 3))
    model.summary()
    
    # Test with dummy data
    test_input = tf.random.normal((5, 28, 28, 3))
    output = model(test_input, training=False)
    print(f"Model output shape: {output.shape}")
    
    return model


def demonstrate_spectral_normalization():
    """Demonstrate spectral normalization wrapper."""
    print("\n=== Spectral Normalization ===")
    
    # Create a layer with spectral normalization
    dense_layer = layers.Dense(64, activation='relu')
    spec_norm_dense = SpectralNormalization(dense_layer)
    
    # Build and test
    x = tf.random.normal((10, 32))
    output = spec_norm_dense(x)
    print(f"Spectral normalized dense output shape: {output.shape}")
    
    # Create a model with spectral normalization
    model = keras.Sequential([
        SpectralNormalization(layers.Dense(128, activation='relu')),
        SpectralNormalization(layers.Dense(64, activation='relu')),
        layers.Dense(10, activation='softmax')
    ])
    
    model.build((None, 32))
    model.summary()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print()
    
    demonstrate_custom_layers()
    create_model_with_custom_layers()
    demonstrate_spectral_normalization()