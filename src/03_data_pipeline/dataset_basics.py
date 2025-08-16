"""
TensorFlow Data Pipeline Basics

This module demonstrates tf.data API for efficient data pipelines.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def demonstrate_dataset_creation():
    """Demonstrate various ways to create tf.data.Dataset objects."""
    print("=== Dataset Creation ===")
    
    # From numpy arrays
    x = np.arange(10)
    y = np.arange(10, 20)
    
    dataset_from_arrays = tf.data.Dataset.from_tensor_slices((x, y))
    print("1. Dataset from numpy arrays:")
    for batch in dataset_from_arrays.take(3):
        print(f"  {batch}")
    
    # From Python lists
    list_data = [1, 2, 3, 4, 5]
    dataset_from_list = tf.data.Dataset.from_tensor_slices(list_data)
    print("\n2. Dataset from Python list:")
    for element in dataset_from_list.take(3):
        print(f"  {element.numpy()}")
    
    # From generator function
    def data_generator():
        for i in range(5):
            yield i, i**2
    
    dataset_from_generator = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=(), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    print("\n3. Dataset from generator:")
    for x, y in dataset_from_generator:
        print(f"  x: {x.numpy()}, y: {y.numpy()}")
    
    # Range dataset
    range_dataset = tf.data.Dataset.range(10)
    print("\n4. Range dataset:")
    for element in range_dataset.take(5):
        print(f"  {element.numpy()}")
    
    # Zip datasets
    dataset1 = tf.data.Dataset.range(5)
    dataset2 = tf.data.Dataset.range(5, 10)
    zipped_dataset = tf.data.Dataset.zip((dataset1, dataset2))
    print("\n5. Zipped datasets:")
    for x, y in zipped_dataset:
        print(f"  ({x.numpy()}, {y.numpy()})")


def demonstrate_dataset_transformations():
    """Demonstrate common dataset transformations."""
    print("\n=== Dataset Transformations ===")
    
    # Create base dataset
    dataset = tf.data.Dataset.range(20)
    
    # Map transformation
    squared_dataset = dataset.map(lambda x: x**2)
    print("1. Map (square each element):")
    for element in squared_dataset.take(5):
        print(f"  {element.numpy()}")
    
    # Filter transformation
    even_dataset = dataset.filter(lambda x: x % 2 == 0)
    print("\n2. Filter (even numbers only):")
    for element in even_dataset.take(5):
        print(f"  {element.numpy()}")
    
    # Batch transformation
    batched_dataset = dataset.batch(4)
    print("\n3. Batch (batch size 4):")
    for batch in batched_dataset.take(2):
        print(f"  {batch.numpy()}")
    
    # Shuffle transformation
    shuffled_dataset = dataset.shuffle(buffer_size=10)
    print("\n4. Shuffle (buffer size 10):")
    for element in shuffled_dataset.take(8):
        print(f"  {element.numpy()}")
    
    # Repeat transformation
    repeated_dataset = dataset.take(3).repeat(2)
    print("\n5. Repeat (repeat 2 times):")
    for element in repeated_dataset:
        print(f"  {element.numpy()}")
    
    # Skip and take
    skipped_dataset = dataset.skip(5).take(3)
    print("\n6. Skip and take:")
    for element in skipped_dataset:
        print(f"  {element.numpy()}")


def demonstrate_advanced_transformations():
    """Demonstrate advanced dataset transformations."""
    print("\n=== Advanced Transformations ===")
    
    # FlatMap transformation
    dataset = tf.data.Dataset.from_tensor_slices([[1, 2], [3, 4], [5, 6]])
    flattened = dataset.flat_map(tf.data.Dataset.from_tensor_slices)
    print("1. FlatMap (flatten nested structure):")
    for element in flattened:
        print(f"  {element.numpy()}")
    
    # Reduce transformation
    numbers = tf.data.Dataset.range(10)
    sum_result = numbers.reduce(0, lambda state, value: state + value)
    print(f"\n2. Reduce (sum): {sum_result.numpy()}")
    
    # Window transformation
    windowed = numbers.window(size=3, shift=1, drop_remainder=True)
    windowed_flat = windowed.flat_map(lambda x: x.batch(3))
    print("\n3. Window (sliding window of size 3):")
    for window in windowed_flat.take(5):
        print(f"  {window.numpy()}")
    
    # Group by key
    pairs = tf.data.Dataset.from_tensor_slices([(1, 'a'), (2, 'b'), (1, 'c'), (2, 'd')])
    
    def key_func(x, y):
        return x
    
    def reduce_func(key, dataset):
        return dataset.batch(10)
    
    grouped = pairs.group_by_window(key_func, reduce_func, window_size=10)
    print("\n4. Group by window:")
    for group in grouped:
        print(f"  {group.numpy()}")


def demonstrate_dataset_performance():
    """Demonstrate performance optimization techniques."""
    print("\n=== Performance Optimization ===")
    
    # Create a dataset
    dataset = tf.data.Dataset.range(1000)
    
    # Cache the dataset
    cached_dataset = dataset.cache()
    print("1. Caching dataset for reuse")
    
    # Prefetch for pipeline efficiency
    prefetched_dataset = dataset.prefetch(tf.data.AUTOTUNE)
    print("2. Prefetching with AUTOTUNE")
    
    # Parallel map
    def slow_function(x):
        # Simulate slow computation
        tf.py_function(lambda: tf.py_function(lambda: None, [], []), [], [])
        return x * x
    
    parallel_mapped = dataset.map(
        slow_function, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print("3. Parallel map processing")
    
    # Interleave for reading multiple files
    def make_dataset(index):
        return tf.data.Dataset.range(index, index + 5)
    
    indices = tf.data.Dataset.range(5)
    interleaved = indices.interleave(
        make_dataset,
        cycle_length=2,
        num_parallel_calls=tf.data.AUTOTUNE
    )
    print("\n4. Interleave (reading from multiple sources):")
    for element in interleaved.take(10):
        print(f"  {element.numpy()}")
    
    # Complete optimization pipeline
    optimized_dataset = (dataset
                        .map(lambda x: x**2, num_parallel_calls=tf.data.AUTOTUNE)
                        .cache()
                        .shuffle(1000)
                        .batch(32)
                        .prefetch(tf.data.AUTOTUNE))
    
    print("\n5. Complete optimization pipeline created")


def demonstrate_dataset_splitting():
    """Demonstrate train/validation/test splitting."""
    print("\n=== Dataset Splitting ===")
    
    # Create dataset
    dataset = tf.data.Dataset.range(1000)
    dataset = dataset.shuffle(1000, seed=42)
    
    # Split ratios
    train_size = int(0.7 * 1000)
    val_size = int(0.2 * 1000)
    test_size = int(0.1 * 1000)
    
    # Split datasets
    train_dataset = dataset.take(train_size)
    remaining = dataset.skip(train_size)
    val_dataset = remaining.take(val_size)
    test_dataset = remaining.skip(val_size)
    
    print(f"Train dataset size: {len(list(train_dataset))}")
    print(f"Validation dataset size: {len(list(val_dataset))}")
    print(f"Test dataset size: {len(list(test_dataset))}")
    
    # Verify no overlap
    train_elements = set(element.numpy() for element in train_dataset)
    val_elements = set(element.numpy() for element in val_dataset)
    test_elements = set(element.numpy() for element in test_dataset)
    
    print(f"Train-Val overlap: {len(train_elements & val_elements)}")
    print(f"Train-Test overlap: {len(train_elements & test_elements)}")
    print(f"Val-Test overlap: {len(val_elements & test_elements)}")


def demonstrate_text_dataset():
    """Demonstrate text data processing."""
    print("\n=== Text Data Processing ===")
    
    # Sample text data
    texts = [
        "Hello world this is TensorFlow",
        "Machine learning is awesome",
        "Deep learning with neural networks",
        "Natural language processing rocks"
    ]
    
    labels = [0, 1, 1, 0]  # Binary classification
    
    # Create dataset
    text_dataset = tf.data.Dataset.from_tensor_slices((texts, labels))
    
    # Text preprocessing
    def preprocess_text(text, label):
        # Convert to lowercase
        text = tf.strings.lower(text)
        # Split into words
        words = tf.strings.split(text)
        return words, label
    
    processed_dataset = text_dataset.map(preprocess_text)
    
    print("Processed text data:")
    for words, label in processed_dataset:
        print(f"  Words: {[w.numpy().decode() for w in words]}, Label: {label.numpy()}")
    
    # Tokenization example
    vocab = ["hello", "world", "this", "is", "tensorflow", "machine", "learning", 
             "awesome", "deep", "with", "neural", "networks", "natural", 
             "language", "processing", "rocks"]
    
    # Create lookup table
    lookup_table = tf.lookup.StaticVocabularyTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab,
            values=tf.range(len(vocab), dtype=tf.int64)
        ),
        num_oov_buckets=1
    )
    
    def tokenize(words, label):
        tokens = lookup_table.lookup(words)
        return tokens, label
    
    tokenized_dataset = processed_dataset.map(tokenize)
    
    print("\nTokenized text data:")
    for tokens, label in tokenized_dataset:
        print(f"  Tokens: {tokens.numpy()}, Label: {label.numpy()}")


def demonstrate_image_dataset():
    """Demonstrate image data processing with dummy data."""
    print("\n=== Image Data Processing ===")
    
    # Create dummy image data
    def create_dummy_image():
        return tf.random.uniform([32, 32, 3], dtype=tf.float32)
    
    def create_dummy_label():
        return tf.random.uniform([], 0, 10, dtype=tf.int32)
    
    # Create dataset of dummy images
    image_dataset = tf.data.Dataset.from_generator(
        lambda: ((create_dummy_image(), create_dummy_label()) for _ in range(100)),
        output_signature=(
            tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Image preprocessing
    def preprocess_image(image, label):
        # Normalize to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        return image, label
    
    preprocessed_dataset = image_dataset.map(
        preprocess_image, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Batch and prefetch
    batched_dataset = (preprocessed_dataset
                      .batch(16)
                      .prefetch(tf.data.AUTOTUNE))
    
    print("Image dataset created and preprocessed")
    
    # Check one batch
    for images, labels in batched_dataset.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")


def demonstrate_dataset_serialization():
    """Demonstrate saving and loading datasets."""
    print("\n=== Dataset Serialization ===")
    
    # Create dataset
    dataset = tf.data.Dataset.range(100)
    dataset = dataset.map(lambda x: x**2)
    dataset = dataset.batch(10)
    
    # Save dataset
    tf.data.Dataset.save(dataset, "/tmp/my_dataset")
    print("Dataset saved to /tmp/my_dataset")
    
    # Load dataset
    loaded_dataset = tf.data.Dataset.load("/tmp/my_dataset")
    print("Dataset loaded from /tmp/my_dataset")
    
    # Verify they're the same
    original_data = list(dataset.take(2))
    loaded_data = list(loaded_dataset.take(2))
    
    print("Original and loaded datasets match:", 
          all(tf.reduce_all(a == b) for a, b in zip(original_data, loaded_data)))


def create_mnist_like_pipeline():
    """Create a complete MNIST-like data pipeline."""
    print("\n=== Complete MNIST-like Pipeline ===")
    
    # Create dummy MNIST-like data
    def create_mnist_data():
        images = tf.random.normal([1000, 28, 28, 1])
        labels = tf.random.uniform([1000], 0, 10, dtype=tf.int32)
        return images, labels
    
    images, labels = create_mnist_data()
    
    # Create dataset
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Preprocessing function
    @tf.function
    def preprocess(image, label):
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Data augmentation
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # One-hot encode labels
        label = tf.one_hot(label, depth=10)
        
        return image, label
    
    # Build complete pipeline
    pipeline = (dataset
               .shuffle(1000, seed=42)
               .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
               .batch(32)
               .cache()
               .prefetch(tf.data.AUTOTUNE))
    
    print("Complete MNIST-like pipeline created")
    
    # Test the pipeline
    for batch_images, batch_labels in pipeline.take(1):
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Image range: [{tf.reduce_min(batch_images):.3f}, {tf.reduce_max(batch_images):.3f}]")
        print(f"Label sum (should be 1.0): {tf.reduce_sum(batch_labels[0]):.1f}")
    
    return pipeline


def run_all_demonstrations():
    """Run all data pipeline demonstrations."""
    demonstrate_dataset_creation()
    demonstrate_dataset_transformations()
    demonstrate_advanced_transformations()
    demonstrate_dataset_performance()
    demonstrate_dataset_splitting()
    demonstrate_text_dataset()
    demonstrate_image_dataset()
    demonstrate_dataset_serialization()
    create_mnist_like_pipeline()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print()
    
    run_all_demonstrations()