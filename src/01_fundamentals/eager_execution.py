"""
TensorFlow Eager Execution and tf.function

This module demonstrates eager execution vs graph execution in TensorFlow.
"""

import tensorflow as tf
import time
import numpy as np


def demonstrate_eager_execution():
    """Demonstrate eager execution in TensorFlow."""
    print("=== Eager Execution ===")
    print(f"Eager execution enabled: {tf.executing_eagerly()}")
    
    # In eager mode, operations execute immediately
    a = tf.constant([1, 2, 3])
    b = tf.constant([4, 5, 6])
    
    print(f"a = {a}")
    print(f"b = {b}")
    
    # Operations execute immediately and return concrete values
    c = a + b
    print(f"a + b = {c}")
    print(f"Result is available immediately: {c.numpy()}")
    
    # Control flow works naturally
    if tf.reduce_sum(c) > 10:
        print("Sum is greater than 10")
    
    # Debugging is easier
    for i, val in enumerate(c):
        print(f"Element {i}: {val.numpy()}")


@tf.function
def simple_graph_function(x, y):
    """A simple function decorated with tf.function."""
    return x * y + 2


@tf.function
def complex_computation(x):
    """More complex computation to demonstrate graph optimization."""
    result = x
    for i in range(10):
        result = result * 2 + 1
    return result


def demonstrate_tf_function():
    """Demonstrate tf.function for graph execution."""
    print("\n=== tf.function Demonstration ===")
    
    # Regular Python function
    def python_function(x, y):
        return x * y + 2
    
    # Test with simple operations
    a = tf.constant(3.0)
    b = tf.constant(4.0)
    
    # Eager execution
    eager_result = python_function(a, b)
    print(f"Eager result: {eager_result}")
    
    # Graph execution
    graph_result = simple_graph_function(a, b)
    print(f"Graph result: {graph_result}")
    
    # Both should give the same result
    print(f"Results are equal: {tf.reduce_all(eager_result == graph_result)}")


def benchmark_execution_modes():
    """Benchmark eager vs graph execution."""
    print("\n=== Performance Comparison ===")
    
    # Create test data
    x = tf.random.normal([1000])
    
    # Eager execution benchmark
    def eager_computation(x):
        result = x
        for i in range(100):
            result = result * 2 + 1
        return result
    
    # Warm up
    for _ in range(3):
        eager_computation(x)
        complex_computation(x)
    
    # Benchmark eager execution
    start_time = time.time()
    for _ in range(10):
        eager_result = eager_computation(x)
    eager_time = time.time() - start_time
    
    # Benchmark graph execution
    start_time = time.time()
    for _ in range(10):
        graph_result = complex_computation(x)
    graph_time = time.time() - start_time
    
    print(f"Eager execution time: {eager_time:.4f} seconds")
    print(f"Graph execution time: {graph_time:.4f} seconds")
    print(f"Speedup: {eager_time / graph_time:.2f}x")
    
    # Verify results are the same
    print(f"Results are close: {tf.reduce_all(tf.abs(eager_result - graph_result) < 1e-6)}")


@tf.function
def conditional_function(x):
    """Demonstrate control flow in tf.function."""
    if tf.reduce_sum(x) > 0:
        return x * 2
    else:
        return x / 2


@tf.function
def loop_function(n):
    """Demonstrate loops in tf.function."""
    result = tf.constant(0.0)
    for i in tf.range(n):
        result += tf.cast(i, tf.float32) ** 2
    return result


def demonstrate_control_flow():
    """Demonstrate control flow in tf.function."""
    print("\n=== Control Flow in tf.function ===")
    
    # Conditional execution
    positive_tensor = tf.constant([1.0, 2.0, 3.0])
    negative_tensor = tf.constant([-1.0, -2.0, -3.0])
    
    pos_result = conditional_function(positive_tensor)
    neg_result = conditional_function(negative_tensor)
    
    print(f"Positive input {positive_tensor.numpy()} -> {pos_result.numpy()}")
    print(f"Negative input {negative_tensor.numpy()} -> {neg_result.numpy()}")
    
    # Loop execution
    n = tf.constant(5)
    loop_result = loop_function(n)
    print(f"Sum of squares from 0 to {n.numpy()-1}: {loop_result.numpy()}")


@tf.function
def polymorphic_function(x):
    """Function that works with different input shapes."""
    return tf.reduce_sum(x, axis=-1)


def demonstrate_polymorphic_functions():
    """Demonstrate polymorphic tf.functions."""
    print("\n=== Polymorphic Functions ===")
    
    # tf.function can handle different input shapes
    x1 = tf.constant([[1, 2], [3, 4]])  # 2x2
    x2 = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # 3x3
    
    result1 = polymorphic_function(x1)
    result2 = polymorphic_function(x2)
    
    print(f"Input shape {x1.shape} -> Output: {result1.numpy()}")
    print(f"Input shape {x2.shape} -> Output: {result2.numpy()}")
    
    # Check concrete functions created
    print(f"Concrete functions created: {len(polymorphic_function._list_all_concrete_functions_for_serialization())}")


@tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
def fixed_signature_function(x):
    """Function with fixed input signature."""
    return tf.reduce_mean(x, axis=0)


def demonstrate_input_signatures():
    """Demonstrate tf.function with input signatures."""
    print("\n=== Input Signatures ===")
    
    # This function only accepts tensors with shape [None, 2]
    x1 = tf.constant([[1.0, 2.0], [3.0, 4.0]])  # Valid: [2, 2]
    x2 = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Valid: [3, 2]
    
    result1 = fixed_signature_function(x1)
    result2 = fixed_signature_function(x2)
    
    print(f"Input shape {x1.shape} -> Output: {result1.numpy()}")
    print(f"Input shape {x2.shape} -> Output: {result2.numpy()}")
    
    # This would fail:
    # x3 = tf.constant([[1.0, 2.0, 3.0]])  # Invalid: [1, 3]
    # result3 = fixed_signature_function(x3)  # Would raise an error


def demonstrate_graph_inspection():
    """Demonstrate how to inspect tf.function graphs."""
    print("\n=== Graph Inspection ===")
    
    @tf.function
    def simple_func(x):
        return x * 2 + 1
    
    # Get the concrete function
    x = tf.constant([1.0, 2.0, 3.0])
    concrete_func = simple_func.get_concrete_function(x)
    
    print(f"Function graph: {concrete_func.graph}")
    print(f"Graph operations:")
    for op in concrete_func.graph.get_operations():
        print(f"  {op.name}: {op.type}")


def run_all_demonstrations():
    """Run all eager execution and tf.function demonstrations."""
    demonstrate_eager_execution()
    demonstrate_tf_function()
    benchmark_execution_modes()
    demonstrate_control_flow()
    demonstrate_polymorphic_functions()
    demonstrate_input_signatures()
    demonstrate_graph_inspection()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print()
    
    run_all_demonstrations()