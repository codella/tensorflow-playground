"""
Performance Optimization: Prefetch and Cache

This module demonstrates performance optimization techniques for tf.data pipelines.
"""

import tensorflow as tf
import numpy as np
import time
from pathlib import Path


def demonstrate_basic_performance_concepts():
    """Demonstrate basic performance concepts in data pipelines."""
    print("=== Basic Performance Concepts ===")
    
    # Create a dataset with artificial delay
    def slow_function(x):
        # Simulate slow I/O or computation
        tf.py_function(lambda: time.sleep(0.01), [], [])
        return x ** 2
    
    # Dataset without optimization
    dataset_slow = tf.data.Dataset.range(50)
    dataset_slow = dataset_slow.map(slow_function)
    
    # Dataset with parallel processing
    dataset_parallel = tf.data.Dataset.range(50)
    dataset_parallel = dataset_parallel.map(
        slow_function, 
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Benchmark
    def benchmark_dataset(dataset, name):
        start_time = time.time()
        for _ in dataset:
            pass
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.2f} seconds")
    
    print("Benchmarking datasets:")
    benchmark_dataset(dataset_slow, "Sequential processing")
    benchmark_dataset(dataset_parallel, "Parallel processing")


def demonstrate_caching():
    """Demonstrate dataset caching for performance."""
    print("\n=== Dataset Caching ===")
    
    # Create a dataset with expensive computation
    def expensive_computation(x):
        # Simulate expensive computation
        result = x
        for _ in range(100):
            result = tf.sin(result) + tf.cos(result)
        return result
    
    # Dataset without caching
    dataset_no_cache = tf.data.Dataset.range(100)
    dataset_no_cache = dataset_no_cache.map(expensive_computation)
    
    # Dataset with memory caching
    dataset_memory_cache = tf.data.Dataset.range(100)
    dataset_memory_cache = dataset_memory_cache.map(expensive_computation)
    dataset_memory_cache = dataset_memory_cache.cache()
    
    # Dataset with file caching
    cache_file = "/tmp/dataset_cache"
    dataset_file_cache = tf.data.Dataset.range(100)
    dataset_file_cache = dataset_file_cache.map(expensive_computation)
    dataset_file_cache = dataset_file_cache.cache(cache_file)
    
    def time_dataset_iterations(dataset, name, iterations=2):
        """Time multiple iterations through the dataset."""
        times = []
        for i in range(iterations):
            start_time = time.time()
            for _ in dataset:
                pass
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"{name} - Iteration {i+1}: {times[-1]:.2f} seconds")
        return times
    
    print("Benchmarking caching (2 iterations each):")
    
    # No cache
    print("\n1. No caching:")
    no_cache_times = time_dataset_iterations(dataset_no_cache, "No cache")
    
    # Memory cache
    print("\n2. Memory caching:")
    memory_cache_times = time_dataset_iterations(dataset_memory_cache, "Memory cache")
    
    # File cache
    print("\n3. File caching:")
    file_cache_times = time_dataset_iterations(dataset_file_cache, "File cache")
    
    # Show speedup
    if len(no_cache_times) > 1 and len(memory_cache_times) > 1:
        speedup = no_cache_times[1] / memory_cache_times[1]
        print(f"\nMemory cache speedup on 2nd iteration: {speedup:.1f}x")


def demonstrate_prefetching():
    """Demonstrate prefetching for pipeline efficiency."""
    print("\n=== Prefetching ===")
    
    def create_training_step_simulation():
        """Simulate a training step that takes time."""
        @tf.function
        def training_step(batch):
            # Simulate GPU computation
            tf.py_function(lambda: time.sleep(0.02), [], [])
            return tf.reduce_sum(batch)
        return training_step
    
    training_step = create_training_step_simulation()
    
    # Dataset without prefetching
    dataset_no_prefetch = tf.data.Dataset.range(100)
    dataset_no_prefetch = dataset_no_prefetch.batch(10)
    
    # Dataset with prefetching
    dataset_with_prefetch = tf.data.Dataset.range(100)
    dataset_with_prefetch = dataset_with_prefetch.batch(10)
    dataset_with_prefetch = dataset_with_prefetch.prefetch(tf.data.AUTOTUNE)
    
    # Manual prefetch buffer size
    dataset_manual_prefetch = tf.data.Dataset.range(100)
    dataset_manual_prefetch = dataset_manual_prefetch.batch(10)
    dataset_manual_prefetch = dataset_manual_prefetch.prefetch(buffer_size=3)
    
    def benchmark_training(dataset, name):
        """Benchmark training with different prefetch settings."""
        start_time = time.time()
        for batch in dataset:
            training_step(batch)
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.2f} seconds")
    
    print("Benchmarking prefetching during training simulation:")
    benchmark_training(dataset_no_prefetch, "No prefetching")
    benchmark_training(dataset_with_prefetch, "AUTOTUNE prefetching")
    benchmark_training(dataset_manual_prefetch, "Manual prefetch (buffer=3)")


def demonstrate_interleaving():
    """Demonstrate interleaving for reading from multiple sources."""
    print("\n=== Interleaving ===")
    
    # Create multiple datasets simulating different files
    def create_file_dataset(file_id, size=50):
        """Simulate reading from a file."""
        def slow_read(x):
            # Simulate slow file I/O
            tf.py_function(lambda: time.sleep(0.001), [], [])
            return file_id * 1000 + x
        
        return (tf.data.Dataset.range(size)
                .map(slow_read))
    
    # Create datasets for multiple files
    file_datasets = [create_file_dataset(i) for i in range(5)]
    
    # Sequential reading (read one file completely, then next)
    sequential_dataset = file_datasets[0]
    for dataset in file_datasets[1:]:
        sequential_dataset = sequential_dataset.concatenate(dataset)
    
    # Interleaved reading
    files_dataset = tf.data.Dataset.from_tensor_slices(list(range(5)))
    interleaved_dataset = files_dataset.interleave(
        lambda file_id: create_file_dataset(file_id),
        cycle_length=3,  # Read from 3 files concurrently
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    def benchmark_reading(dataset, name):
        start_time = time.time()
        count = 0
        for _ in dataset:
            count += 1
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.2f} seconds ({count} elements)")
    
    print("Benchmarking file reading patterns:")
    benchmark_reading(sequential_dataset, "Sequential reading")
    benchmark_reading(interleaved_dataset, "Interleaved reading")


def demonstrate_optimal_pipeline():
    """Demonstrate an optimally configured data pipeline."""
    print("\n=== Optimal Pipeline Configuration ===")
    
    def create_complex_dataset():
        """Create a dataset with multiple expensive operations."""
        
        def expensive_parse(x):
            # Simulate parsing (e.g., image decoding)
            tf.py_function(lambda: time.sleep(0.005), [], [])
            return tf.cast(x, tf.float32) / 255.0
        
        def expensive_augment(x):
            # Simulate augmentation
            tf.py_function(lambda: time.sleep(0.003), [], [])
            return x * tf.random.uniform([], 0.8, 1.2)
        
        # Base dataset
        dataset = tf.data.Dataset.range(200)
        
        return dataset, expensive_parse, expensive_augment
    
    dataset, parse_fn, augment_fn = create_complex_dataset()
    
    # Naive pipeline
    naive_pipeline = (dataset
                     .map(parse_fn)
                     .map(augment_fn)
                     .batch(16))
    
    # Optimized pipeline
    optimized_pipeline = (dataset
                         .map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
                         .cache()  # Cache after parsing
                         .map(augment_fn, num_parallel_calls=tf.data.AUTOTUNE)
                         .batch(16)
                         .prefetch(tf.data.AUTOTUNE))
    
    # Further optimized with fusion
    fusion_optimized_pipeline = (dataset
                                .map(lambda x: augment_fn(parse_fn(x)),
                                     num_parallel_calls=tf.data.AUTOTUNE)
                                .cache()
                                .batch(16)
                                .prefetch(tf.data.AUTOTUNE))
    
    def benchmark_pipeline(pipeline, name):
        start_time = time.time()
        for _ in pipeline:
            pass
        end_time = time.time()
        print(f"{name}: {end_time - start_time:.2f} seconds")
    
    print("Benchmarking pipeline optimizations:")
    benchmark_pipeline(naive_pipeline, "Naive pipeline")
    benchmark_pipeline(optimized_pipeline, "Optimized pipeline")
    benchmark_pipeline(fusion_optimized_pipeline, "Fusion optimized pipeline")


def analyze_pipeline_performance():
    """Analyze and profile pipeline performance."""
    print("\n=== Pipeline Performance Analysis ===")
    
    # Create a dataset with identifiable bottlenecks
    def cpu_intensive_task(x):
        # Simulate CPU-intensive task
        result = x
        for _ in tf.range(50):
            result = tf.sin(result) + tf.cos(result)
        return result
    
    def io_intensive_task(x):
        # Simulate I/O intensive task
        tf.py_function(lambda: time.sleep(0.001), [], [])
        return x * 2
    
    # Create different pipeline configurations
    configurations = {
        "baseline": lambda ds: ds.map(lambda x: cpu_intensive_task(io_intensive_task(x))),
        
        "parallel_map": lambda ds: ds.map(
            lambda x: cpu_intensive_task(io_intensive_task(x)),
            num_parallel_calls=tf.data.AUTOTUNE
        ),
        
        "separate_parallel": lambda ds: (ds
            .map(io_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
            .map(cpu_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
        ),
        
        "with_cache": lambda ds: (ds
            .map(io_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .map(cpu_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
        ),
        
        "full_optimization": lambda ds: (ds
            .map(io_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
            .cache()
            .map(cpu_intensive_task, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(8)
            .prefetch(tf.data.AUTOTUNE)
        )
    }
    
    base_dataset = tf.data.Dataset.range(50)
    
    print("Pipeline performance comparison:")
    baseline_time = None
    
    for name, config_fn in configurations.items():
        dataset = config_fn(base_dataset)
        
        start_time = time.time()
        for _ in dataset:
            pass
        end_time = time.time()
        
        elapsed = end_time - start_time
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        print(f"{name:20}: {elapsed:.2f}s (speedup: {speedup:.1f}x)")


def demonstrate_memory_efficiency():
    """Demonstrate memory-efficient data loading."""
    print("\n=== Memory Efficiency ===")
    
    # Large dataset simulation
    def create_large_dataset_simulation():
        """Simulate a large dataset that doesn't fit in memory."""
        
        # Generator that yields data on demand
        def data_generator():
            for i in range(1000):  # Simulate 1000 large samples
                # Simulate large data (e.g., high-res image)
                large_data = tf.random.normal([512, 512, 3])
                label = i % 10
                yield large_data, label
        
        return tf.data.Dataset.from_generator(
            data_generator,
            output_signature=(
                tf.TensorSpec(shape=[512, 512, 3], dtype=tf.float32),
                tf.TensorSpec(shape=(), dtype=tf.int32)
            )
        )
    
    # Memory-efficient pipeline
    large_dataset = create_large_dataset_simulation()
    
    # Efficient processing: only keep small batches in memory
    efficient_pipeline = (large_dataset
                         .map(lambda x, y: (tf.image.resize(x, [224, 224]), y),
                              num_parallel_calls=tf.data.AUTOTUNE)
                         .batch(4)  # Small batch size
                         .prefetch(2))  # Small prefetch buffer
    
    print("Memory-efficient pipeline created")
    print("Processing large dataset with minimal memory footprint...")
    
    # Process a few batches
    batch_count = 0
    for batch_data, batch_labels in efficient_pipeline.take(3):
        batch_count += 1
        print(f"Batch {batch_count}: "
              f"Data shape {batch_data.shape}, "
              f"Memory usage minimal due to streaming")


def create_production_pipeline():
    """Create a production-ready data pipeline with all optimizations."""
    print("\n=== Production Pipeline ===")
    
    @tf.function
    def preprocess_function(image, label):
        """Preprocessing function with augmentation."""
        # Normalize
        image = tf.cast(image, tf.float32) / 255.0
        
        # Random augmentation
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_brightness(image, max_delta=0.1)
        
        # One-hot encode labels
        label = tf.one_hot(label, depth=10)
        
        return image, label
    
    def create_production_dataset(data_dir="/tmp/dummy_data", batch_size=32):
        """Create a production-ready dataset pipeline."""
        
        # Simulate file paths
        file_paths = [f"/tmp/image_{i}.jpg" for i in range(1000)]
        labels = tf.random.uniform([1000], 0, 10, dtype=tf.int32)
        
        # Create dataset from file paths
        dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
        
        # Shuffle with large buffer
        dataset = dataset.shuffle(buffer_size=1000, seed=42)
        
        # Parse and preprocess (simulated)
        def parse_function(filename, label):
            # In practice, this would decode the image file
            # For demo, create a random image
            image = tf.random.uniform([224, 224, 3], dtype=tf.float32)
            return image, label
        
        # Build optimized pipeline
        dataset = (dataset
                  .map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
                  .cache()  # Cache after expensive parsing
                  .map(preprocess_function, num_parallel_calls=tf.data.AUTOTUNE)
                  .batch(batch_size)
                  .prefetch(tf.data.AUTOTUNE))
        
        return dataset
    
    # Create and test production pipeline
    prod_dataset = create_production_dataset()
    
    print("Production pipeline features:")
    print("✓ Parallel data loading")
    print("✓ Caching after expensive operations")
    print("✓ Parallel preprocessing")
    print("✓ Batching")
    print("✓ Prefetching with AUTOTUNE")
    print("✓ Random shuffling")
    print("✓ Data augmentation")
    
    # Test the pipeline
    for batch_images, batch_labels in prod_dataset.take(1):
        print(f"\nPipeline output:")
        print(f"Batch images shape: {batch_images.shape}")
        print(f"Batch labels shape: {batch_labels.shape}")
        print(f"Image value range: [{tf.reduce_min(batch_images):.3f}, {tf.reduce_max(batch_images):.3f}]")
    
    return prod_dataset


def run_all_demonstrations():
    """Run all performance optimization demonstrations."""
    demonstrate_basic_performance_concepts()
    demonstrate_caching()
    demonstrate_prefetching()
    demonstrate_interleaving()
    demonstrate_optimal_pipeline()
    analyze_pipeline_performance()
    demonstrate_memory_efficiency()
    create_production_pipeline()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print()
    
    run_all_demonstrations()