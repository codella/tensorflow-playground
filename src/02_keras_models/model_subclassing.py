"""
Keras Model Subclassing Examples

This module demonstrates model subclassing for creating custom models.
"""

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


class SimpleCustomModel(keras.Model):
    """A simple custom model using subclassing."""
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.dense1 = layers.Dense(64, activation='relu')
        self.dense2 = layers.Dense(32, activation='relu')
        self.dropout = layers.Dropout(0.3)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        if training:
            x = self.dropout(x, training=training)
        return self.classifier(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({"num_classes": self.num_classes})
        return config


class ResidualBlock(layers.Layer):
    """A custom residual block layer."""
    
    def __init__(self, filters, kernel_size=3, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        
        self.conv1 = layers.Conv2D(filters, kernel_size, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filters, kernel_size, padding='same')
        self.bn2 = layers.BatchNormalization()
        self.activation = layers.Activation('relu')
        self.add = layers.Add()
    
    def build(self, input_shape):
        super().build(input_shape)
        # Add projection if input channels != output channels
        if input_shape[-1] != self.filters:
            self.projection = layers.Conv2D(self.filters, 1, padding='same')
        else:
            self.projection = None
    
    def call(self, inputs, training=None):
        # Save input for skip connection
        shortcut = inputs
        
        # Main path
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        # Adjust shortcut if needed
        if self.projection is not None:
            shortcut = self.projection(shortcut)
        
        # Add skip connection and apply activation
        x = self.add([x, shortcut])
        return self.activation(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config


class CustomResNet(keras.Model):
    """A custom ResNet implementation."""
    
    def __init__(self, num_classes=10, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        
        self.conv1 = layers.Conv2D(64, 7, padding='same', activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPooling2D(3, strides=2, padding='same')
        
        # Residual blocks
        self.res_block1 = ResidualBlock(64)
        self.res_block2 = ResidualBlock(64)
        self.res_block3 = ResidualBlock(128)
        self.res_block4 = ResidualBlock(128)
        
        self.global_pool = layers.GlobalAveragePooling2D()
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.pool1(x)
        
        x = self.res_block1(x, training=training)
        x = self.res_block2(x, training=training)
        x = self.res_block3(x, training=training)
        x = self.res_block4(x, training=training)
        
        x = self.global_pool(x)
        return self.classifier(x)


class AttentionLayer(layers.Layer):
    """A custom attention layer."""
    
    def __init__(self, units, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.W = layers.Dense(units, use_bias=False)
        self.U = layers.Dense(units, use_bias=False)
        self.V = layers.Dense(1, use_bias=False)
    
    def call(self, hidden_states):
        # hidden_states shape: (batch_size, time_steps, hidden_size)
        
        # Compute attention scores
        score = self.V(tf.nn.tanh(self.W(hidden_states)))
        
        # Apply softmax to get attention weights
        attention_weights = tf.nn.softmax(score, axis=1)
        
        # Apply attention weights to hidden states
        context_vector = tf.reduce_sum(attention_weights * hidden_states, axis=1)
        
        return context_vector, attention_weights


class TextClassifierWithAttention(keras.Model):
    """A text classifier with attention mechanism."""
    
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, num_classes=2, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.lstm_units = lstm_units
        self.num_classes = num_classes
        
        self.embedding = layers.Embedding(vocab_size, embedding_dim)
        self.lstm = layers.LSTM(lstm_units, return_sequences=True, dropout=0.2)
        self.attention = AttentionLayer(lstm_units)
        self.dropout = layers.Dropout(0.5)
        self.classifier = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=None):
        x = self.embedding(inputs)
        lstm_output = self.lstm(x, training=training)
        
        # Apply attention
        context_vector, attention_weights = self.attention(lstm_output)
        
        if training:
            context_vector = self.dropout(context_vector, training=training)
        
        return self.classifier(context_vector)


class VariationalAutoencoder(keras.Model):
    """A Variational Autoencoder implementation."""
    
    def __init__(self, latent_dim=32, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder_dense1 = layers.Dense(256, activation='relu')
        self.encoder_dense2 = layers.Dense(128, activation='relu')
        self.mean_layer = layers.Dense(latent_dim)
        self.log_var_layer = layers.Dense(latent_dim)
        
        # Decoder
        self.decoder_dense1 = layers.Dense(128, activation='relu')
        self.decoder_dense2 = layers.Dense(256, activation='relu')
        self.decoder_output = layers.Dense(784, activation='sigmoid')
    
    def encode(self, x):
        x = self.encoder_dense1(x)
        x = self.encoder_dense2(x)
        mean = self.mean_layer(x)
        log_var = self.log_var_layer(x)
        return mean, log_var
    
    def reparameterize(self, mean, log_var):
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon
    
    def decode(self, z):
        x = self.decoder_dense1(z)
        x = self.decoder_dense2(x)
        return self.decoder_output(x)
    
    def call(self, inputs, training=None):
        mean, log_var = self.encode(inputs)
        z = self.reparameterize(mean, log_var)
        reconstructed = self.decode(z)
        
        # Add KL divergence loss during training
        if training:
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var)
            )
            self.add_loss(kl_loss)
        
        return reconstructed


class GAN(keras.Model):
    """A simple Generative Adversarial Network."""
    
    def __init__(self, latent_dim=100, **kwargs):
        super().__init__(**kwargs)
        self.latent_dim = latent_dim
        
        # Generator
        self.generator = self.build_generator()
        
        # Discriminator
        self.discriminator = self.build_discriminator()
    
    def build_generator(self):
        model = keras.Sequential([
            layers.Dense(256, activation='relu', input_shape=(self.latent_dim,)),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(1024, activation='relu'),
            layers.BatchNormalization(),
            layers.Dense(784, activation='tanh')
        ], name='generator')
        return model
    
    def build_discriminator(self):
        model = keras.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation='sigmoid')
        ], name='discriminator')
        return model
    
    def compile(self, g_optimizer, d_optimizer, loss_fn):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.loss_fn = loss_fn
    
    @tf.function
    def train_step(self, real_images):
        batch_size = tf.shape(real_images)[0]
        
        # Generate random noise for generator
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))
        
        # Train discriminator
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            real_logits = self.discriminator(real_images, training=True)
            
            d_cost = self.loss_fn(tf.ones_like(real_logits), real_logits) + \
                     self.loss_fn(tf.zeros_like(fake_logits), fake_logits)
        
        d_gradient = tape.gradient(d_cost, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(
            zip(d_gradient, self.discriminator.trainable_variables)
        )
        
        # Train generator
        with tf.GradientTape() as tape:
            fake_images = self.generator(random_latent_vectors, training=True)
            fake_logits = self.discriminator(fake_images, training=True)
            g_cost = self.loss_fn(tf.ones_like(fake_logits), fake_logits)
        
        g_gradient = tape.gradient(g_cost, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_variables)
        )
        
        return {"d_loss": d_cost, "g_loss": g_cost}


class MultiTaskModel(keras.Model):
    """A multi-task learning model."""
    
    def __init__(self, num_classes_task1=10, num_classes_task2=5, **kwargs):
        super().__init__(**kwargs)
        self.num_classes_task1 = num_classes_task1
        self.num_classes_task2 = num_classes_task2
        
        # Shared layers
        self.shared_dense1 = layers.Dense(128, activation='relu')
        self.shared_dense2 = layers.Dense(64, activation='relu')
        
        # Task-specific layers
        self.task1_dense = layers.Dense(32, activation='relu')
        self.task1_output = layers.Dense(num_classes_task1, activation='softmax', name='task1')
        
        self.task2_dense = layers.Dense(32, activation='relu')
        self.task2_output = layers.Dense(num_classes_task2, activation='softmax', name='task2')
    
    def call(self, inputs, training=None):
        # Shared feature extraction
        x = self.shared_dense1(inputs)
        shared_features = self.shared_dense2(x)
        
        # Task 1 branch
        task1_x = self.task1_dense(shared_features)
        task1_output = self.task1_output(task1_x)
        
        # Task 2 branch
        task2_x = self.task2_dense(shared_features)
        task2_output = self.task2_output(task2_x)
        
        return {'task1': task1_output, 'task2': task2_output}


def demonstrate_model_subclassing():
    """Demonstrate various custom model implementations."""
    print("=== Model Subclassing Demonstrations ===")
    
    # Simple custom model
    print("\n1. Simple Custom Model:")
    simple_model = SimpleCustomModel(num_classes=10)
    simple_model.build((None, 784))
    simple_model.summary()
    
    # Custom ResNet
    print("\n2. Custom ResNet:")
    resnet_model = CustomResNet(num_classes=10)
    resnet_model.build((None, 32, 32, 3))
    resnet_model.summary()
    
    # Text classifier with attention
    print("\n3. Text Classifier with Attention:")
    text_model = TextClassifierWithAttention(vocab_size=10000, num_classes=2)
    text_model.build((None, 100))  # sequence length = 100
    text_model.summary()
    
    # Variational Autoencoder
    print("\n4. Variational Autoencoder:")
    vae_model = VariationalAutoencoder(latent_dim=32)
    vae_model.build((None, 784))
    vae_model.summary()
    
    # Multi-task model
    print("\n5. Multi-task Model:")
    multitask_model = MultiTaskModel(num_classes_task1=10, num_classes_task2=5)
    multitask_model.build((None, 784))
    multitask_model.summary()
    
    # Test models with dummy data
    print("\n6. Testing models with dummy data:")
    
    # Test simple model
    x_test = np.random.random((5, 784))
    pred = simple_model(x_test, training=False)
    print(f"Simple model output shape: {pred.shape}")
    
    # Test ResNet
    x_img = np.random.random((2, 32, 32, 3))
    pred = resnet_model(x_img, training=False)
    print(f"ResNet output shape: {pred.shape}")
    
    # Test text model
    x_text = np.random.randint(0, 10000, (3, 100))
    pred = text_model(x_text, training=False)
    print(f"Text model output shape: {pred.shape}")
    
    # Test VAE
    x_vae = np.random.random((4, 784))
    pred = vae_model(x_vae, training=False)
    print(f"VAE output shape: {pred.shape}")
    
    # Test multi-task model
    x_multi = np.random.random((3, 784))
    pred = multitask_model(x_multi, training=False)
    print(f"Multi-task model outputs: {[f'{k}: {v.shape}' for k, v in pred.items()]}")


def demonstrate_custom_training_loop():
    """Demonstrate custom training loop with subclassed model."""
    print("\n=== Custom Training Loop ===")
    
    # Create model
    model = SimpleCustomModel(num_classes=10)
    
    # Create optimizer and loss function
    optimizer = keras.optimizers.Adam(learning_rate=0.001)
    loss_fn = keras.losses.CategoricalCrossentropy()
    
    # Create dummy data
    x_train = np.random.random((100, 784))
    y_train = keras.utils.to_categorical(np.random.randint(10, size=(100,)), 10)
    
    # Training step function
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            predictions = model(x, training=True)
            loss = loss_fn(y, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        
        return loss
    
    # Training loop
    epochs = 3
    batch_size = 32
    
    print(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(x_train) // batch_size
        
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            
            loss = train_step(batch_x, batch_y)
            epoch_loss += loss
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("Keras version:", keras.__version__)
    print()
    
    demonstrate_model_subclassing()
    demonstrate_custom_training_loop()