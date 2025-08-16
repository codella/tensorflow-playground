"""
TensorFlow Tensor Basics

This module demonstrates fundamental tensor operations in TensorFlow.
"""

import tensorflow as tf
import numpy as np


def demonstrate_tensor_creation():
    """
    Demonstrate various ways to create tensors in TensorFlow.
    
    Tensors are the fundamental data structure in TensorFlow - they are 
    multi-dimensional arrays similar to NumPy arrays but optimized for 
    GPU computation and automatic differentiation.
    """
    print("=== Tensor Creation ===")
    
    # STEP 1: Creating tensors from Python data structures
    # tf.constant() creates immutable tensors from Python data
    
    # Scalar (0-dimensional tensor) - just a single number
    scalar = tf.constant(7)
    
    # Vector (1-dimensional tensor) - a list of numbers
    vector = tf.constant([1, 2, 3, 4])
    
    # Matrix (2-dimensional tensor) - rows and columns of numbers
    matrix = tf.constant([[1, 2], [3, 4], [5, 6]])  # 3 rows, 2 columns
    
    # 3D Tensor - think of it as multiple matrices stacked together
    tensor_3d = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # 2x2x2 shape
    
    print(f"Scalar (0D): {scalar}")
    print(f"Vector (1D): {vector}")
    print(f"Matrix (2D):\n{matrix}")
    print(f"3D Tensor:\n{tensor_3d}")
    
    # STEP 2: Creating tensors with specific shapes and initial values
    # These functions are useful when you need tensors of a certain size
    
    # Create a tensor filled with zeros - common for initialization
    zeros = tf.zeros([3, 4])  # 3 rows, 4 columns, all zeros
    
    # Create a tensor filled with ones - useful for masks or initialization
    ones = tf.ones([2, 3, 4])  # 2x3x4 tensor, all ones
    
    # Create a tensor filled with a specific value
    filled = tf.fill([2, 3], 9)  # 2x3 tensor, all filled with 9
    
    print(f"\nZeros tensor (3x4):\n{zeros}")
    print(f"\nOnes tensor (2x3x4) shape: {ones.shape}")
    print(f"\nFilled tensor (2x3) with 9s:\n{filled}")
    
    # STEP 3: Creating random tensors
    # Random tensors are crucial for neural network weight initialization
    
    # Normal distribution (Gaussian) - most common for weight initialization
    # mean=0.0, stddev=1.0 creates standard normal distribution
    random_normal = tf.random.normal([3, 3], mean=0.0, stddev=1.0)
    
    # Uniform distribution - values evenly distributed in a range
    # minval=0, maxval=10 creates integers from 0 to 9
    random_uniform = tf.random.uniform([2, 4], minval=0, maxval=10, dtype=tf.int32)
    
    print(f"\nRandom normal (3x3):\n{random_normal}")
    print(f"\nRandom uniform integers (2x4):\n{random_uniform}")


def demonstrate_tensor_properties():
    """Demonstrate tensor properties like shape, rank, and dtype."""
    print("\n=== Tensor Properties ===")
    
    tensor = tf.constant([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]])
    
    print(f"Tensor:\n{tensor}")
    print(f"Shape: {tensor.shape}")
    print(f"Rank (number of dimensions): {tf.rank(tensor)}")
    print(f"Size (total number of elements): {tf.size(tensor)}")
    print(f"Data type: {tensor.dtype}")
    
    # Different data types
    float_tensor = tf.constant([1.0, 2.0, 3.0], dtype=tf.float32)
    int_tensor = tf.constant([1, 2, 3], dtype=tf.int32)
    bool_tensor = tf.constant([True, False, True], dtype=tf.bool)
    string_tensor = tf.constant(["hello", "tensorflow"], dtype=tf.string)
    
    print(f"\nFloat tensor: {float_tensor} (dtype: {float_tensor.dtype})")
    print(f"Int tensor: {int_tensor} (dtype: {int_tensor.dtype})")
    print(f"Bool tensor: {bool_tensor} (dtype: {bool_tensor.dtype})")
    print(f"String tensor: {string_tensor} (dtype: {string_tensor.dtype})")


def demonstrate_tensor_operations():
    """
    Demonstrate basic tensor operations.
    
    Understanding tensor operations is crucial for building neural networks.
    Most operations in TensorFlow work element-wise by default.
    """
    print("\n=== Tensor Operations ===")
    
    # STEP 1: Set up example tensors for demonstration
    # Using float32 is common in deep learning for memory efficiency
    a = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    b = tf.constant([[5, 6], [7, 8]], dtype=tf.float32)
    
    print(f"Tensor A:\n{a}")
    print(f"Tensor B:\n{b}")
    
    # STEP 2: Element-wise operations
    # These operations work on corresponding elements of the tensors
    
    # Addition: adds corresponding elements
    print(f"\nA + B (element-wise addition):\n{tf.add(a, b)}")
    # Alternative syntax: a + b (Python operator overloading)
    
    # Subtraction: subtracts corresponding elements  
    print(f"A - B (element-wise subtraction):\n{tf.subtract(a, b)}")
    
    # Multiplication: multiplies corresponding elements (NOT matrix multiplication!)
    print(f"A * B (element-wise multiplication):\n{tf.multiply(a, b)}")
    
    # Division: divides corresponding elements
    print(f"A / B (element-wise division):\n{tf.divide(a, b)}")
    
    # STEP 3: Matrix operations (linear algebra)
    # These follow mathematical rules for matrices
    
    # Matrix multiplication: follows linear algebra rules (rows × columns)
    print(f"\nA @ B (matrix multiplication):\n{tf.matmul(a, b)}")
    # Note: @ operator is equivalent to tf.matmul()
    
    # Transpose: flips rows and columns
    print(f"A transpose (flip rows/columns):\n{tf.transpose(a)}")
    
    # STEP 4: Reduction operations
    # These operations reduce tensor dimensions by aggregating values
    
    c = tf.constant([[1, 2, 3], [4, 5, 6]], dtype=tf.float32)
    print(f"\nTensor C (2x3):\n{c}")
    
    # Reduce along all dimensions (results in scalar)
    print(f"Sum of all elements: {tf.reduce_sum(c)}")
    
    # Reduce along specific axis
    # axis=0: operate along rows (result: [1+4, 2+5, 3+6])
    print(f"Sum along axis 0 (column sums): {tf.reduce_sum(c, axis=0)}")
    
    # axis=1: operate along columns (result: [1+2+3, 4+5+6])
    print(f"Sum along axis 1 (row sums): {tf.reduce_sum(c, axis=1)}")
    
    # Other useful reductions
    print(f"Mean (average): {tf.reduce_mean(c)}")
    print(f"Max (largest value): {tf.reduce_max(c)}")
    print(f"Min (smallest value): {tf.reduce_min(c)}")


def demonstrate_tensor_indexing():
    """Demonstrate tensor indexing and slicing."""
    print("\n=== Tensor Indexing and Slicing ===")
    
    tensor = tf.constant([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    print(f"Original tensor:\n{tensor}")
    
    # Basic indexing
    print(f"\nFirst row: {tensor[0]}")
    print(f"First column: {tensor[:, 0]}")
    print(f"Element at (1, 2): {tensor[1, 2]}")
    
    # Slicing
    print(f"\nFirst 2 rows: \n{tensor[:2]}")
    print(f"Last 2 columns:\n{tensor[:, -2:]}")
    print(f"Every other element in first row: {tensor[0, ::2]}")
    
    # Boolean indexing
    mask = tensor > 6
    print(f"\nMask (elements > 6):\n{mask}")
    print(f"Elements > 6: {tf.boolean_mask(tensor, mask)}")


def demonstrate_tensor_reshaping():
    """Demonstrate tensor reshaping operations."""
    print("\n=== Tensor Reshaping ===")
    
    tensor = tf.constant([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    print(f"Original tensor: {tensor}")
    print(f"Shape: {tensor.shape}")
    
    # Reshape to different dimensions
    reshaped_2d = tf.reshape(tensor, [3, 4])
    print(f"\nReshaped to 3x4:\n{reshaped_2d}")
    
    reshaped_3d = tf.reshape(tensor, [2, 2, 3])
    print(f"\nReshaped to 2x2x3:\n{reshaped_3d}")
    
    # Using -1 for automatic dimension calculation
    auto_reshape = tf.reshape(tensor, [4, -1])
    print(f"\nAuto-reshaped to 4x?:\n{auto_reshape}")
    
    # Expand and squeeze dimensions
    expanded = tf.expand_dims(tensor, axis=0)
    print(f"\nExpanded dimensions (add axis 0): {expanded.shape}")
    
    squeezed = tf.squeeze(expanded)
    print(f"Squeezed back: {squeezed.shape}")


def demonstrate_variables():
    """Demonstrate TensorFlow Variables."""
    print("\n=== TensorFlow Variables ===")
    
    # Creating variables
    var = tf.Variable([1.0, 2.0, 3.0], name="my_variable")
    print(f"Variable: {var}")
    print(f"Value: {var.numpy()}")
    print(f"Trainable: {var.trainable}")
    
    # Modifying variables
    print(f"\nOriginal value: {var.numpy()}")
    var.assign([4.0, 5.0, 6.0])
    print(f"After assign: {var.numpy()}")
    
    var.assign_add([1.0, 1.0, 1.0])
    print(f"After assign_add: {var.numpy()}")
    
    var.assign_sub([0.5, 0.5, 0.5])
    print(f"After assign_sub: {var.numpy()}")
    
    # Variable vs Constant
    constant = tf.constant([1.0, 2.0, 3.0])
    print(f"\nConstant cannot be modified: {constant}")
    
    # Creating variables with different initializers
    normal_var = tf.Variable(tf.random.normal([3, 3]))
    uniform_var = tf.Variable(tf.random.uniform([2, 2]))
    zeros_var = tf.Variable(tf.zeros([4]))
    
    print(f"\nNormal initialized variable shape: {normal_var.shape}")
    print(f"Uniform initialized variable shape: {uniform_var.shape}")
    print(f"Zeros initialized variable: {zeros_var.numpy()}")


def run_all_demonstrations():
    """Run all tensor demonstrations."""
    demonstrate_tensor_creation()
    demonstrate_tensor_properties()
    demonstrate_tensor_operations()
    demonstrate_tensor_indexing()
    demonstrate_tensor_reshaping()
    demonstrate_variables()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print("GPU available:", tf.config.list_physical_devices('GPU'))
    print()
    
    run_all_demonstrations()