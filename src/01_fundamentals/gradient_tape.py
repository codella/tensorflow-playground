"""
TensorFlow Automatic Differentiation with GradientTape

This module demonstrates automatic differentiation using tf.GradientTape.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_basic_gradients():
    """
    Demonstrate basic gradient computation with GradientTape.
    
    Automatic differentiation is the foundation of neural network training.
    GradientTape records operations to compute gradients automatically,
    eliminating the need for manual derivative calculations.
    """
    print("=== Basic Gradient Computation ===")
    
    # STEP 1: Single variable gradient
    # Create a Variable (not a constant!) - only Variables are watched by GradientTape
    x = tf.Variable(3.0)
    print(f"Input variable x = {x.numpy()}")
    
    # STEP 2: Record operations in the tape context
    # Everything inside this context is recorded for gradient computation
    with tf.GradientTape() as tape:
        # Define our function: f(x) = x²
        y = x ** 2
        print(f"Forward pass: f(x) = x² = {y.numpy()}")
    
    # STEP 3: Compute gradient automatically
    # tape.gradient(target, source) computes ∂target/∂source
    gradient = tape.gradient(y, x)
    
    # Mathematical verification: d/dx(x²) = 2x
    expected_gradient = 2 * x.numpy()
    print(f"Computed gradient: df/dx = {gradient.numpy()}")
    print(f"Expected gradient: df/dx = 2x = {expected_gradient}")
    print(f"✓ Gradients match!" if abs(gradient.numpy() - expected_gradient) < 1e-6 else "✗ Error!")
    
    # STEP 4: Multiple variables (partial derivatives)
    print(f"\n--- Multiple Variables ---")
    x1 = tf.Variable(2.0, name='x1')  # Names help with debugging
    x2 = tf.Variable(3.0, name='x2')
    
    with tf.GradientTape() as tape:
        # More complex function: f(x1, x2) = x1² + x2³ + x1*x2
        z = x1 ** 2 + x2 ** 3 + x1 * x2
        print(f"Function: f(x1, x2) = x1² + x2³ + x1*x2")
        print(f"f({x1.numpy()}, {x2.numpy()}) = {z.numpy()}")
    
    # Compute partial derivatives for both variables
    gradients = tape.gradient(z, [x1, x2])
    
    # Mathematical verification:
    # ∂f/∂x1 = 2*x1 + x2 = 2*2 + 3 = 7
    # ∂f/∂x2 = 3*x2² + x1 = 3*9 + 2 = 29
    expected_grad_x1 = 2 * x1.numpy() + x2.numpy()
    expected_grad_x2 = 3 * x2.numpy()**2 + x1.numpy()
    
    print(f"Computed ∂f/∂x1 = {gradients[0].numpy()}")
    print(f"Expected ∂f/∂x1 = 2*x1 + x2 = {expected_grad_x1}")
    print(f"Computed ∂f/∂x2 = {gradients[1].numpy()}")
    print(f"Expected ∂f/∂x2 = 3*x2² + x1 = {expected_grad_x2}")
    
    # Why is this useful?
    # In neural networks, we need gradients to update weights during training.
    # GradientTape automates this for arbitrarily complex functions!


def demonstrate_gradient_of_functions():
    """Demonstrate gradients of common mathematical functions."""
    print("\n=== Gradients of Common Functions ===")
    
    x = tf.Variable(1.0)
    
    functions = [
        ("sin(x)", lambda x: tf.sin(x), "cos(x)"),
        ("cos(x)", lambda x: tf.cos(x), "-sin(x)"),
        ("exp(x)", lambda x: tf.exp(x), "exp(x)"),
        ("log(x)", lambda x: tf.math.log(x), "1/x"),
        ("x^3", lambda x: x**3, "3*x^2"),
    ]
    
    for name, func, expected_grad in functions:
        with tf.GradientTape() as tape:
            y = func(x)
        
        gradient = tape.gradient(y, x)
        print(f"{name}: f({x.numpy()}) = {y.numpy():.4f}, "
              f"f'({x.numpy()}) = {gradient.numpy():.4f} (expected: {expected_grad})")


def demonstrate_higher_order_gradients():
    """Demonstrate higher-order gradients (gradients of gradients)."""
    print("\n=== Higher-Order Gradients ===")
    
    x = tf.Variable(2.0)
    
    # Compute second derivative
    with tf.GradientTape() as tape2:
        with tf.GradientTape() as tape1:
            y = x ** 4  # f(x) = x^4
        
        # First derivative: dy/dx = 4*x^3
        first_grad = tape1.gradient(y, x)
    
    # Second derivative: d²y/dx² = 12*x^2
    second_grad = tape2.gradient(first_grad, x)
    
    print(f"f(x) = x^4, x = {x.numpy()}")
    print(f"f'(x) = {first_grad.numpy()} (expected: {4 * x.numpy()**3})")
    print(f"f''(x) = {second_grad.numpy()} (expected: {12 * x.numpy()**2})")


def demonstrate_gradient_with_neural_network():
    """Demonstrate gradient computation with a simple neural network."""
    print("\n=== Gradients with Neural Network ===")
    
    # Simple linear model: y = W*x + b
    W = tf.Variable([[1.0, 2.0], [3.0, 4.0]], name="weights")
    b = tf.Variable([0.5, -0.5], name="bias")
    
    x = tf.constant([[1.0, 2.0]])  # Input
    y_true = tf.constant([[5.0, 10.0]])  # True output
    
    with tf.GradientTape() as tape:
        # Forward pass
        y_pred = tf.matmul(x, W) + b
        
        # Loss (Mean Squared Error)
        loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    # Compute gradients
    gradients = tape.gradient(loss, [W, b])
    grad_W, grad_b = gradients
    
    print(f"Input: {x.numpy()}")
    print(f"Weights:\n{W.numpy()}")
    print(f"Bias: {b.numpy()}")
    print(f"Predicted: {y_pred.numpy()}")
    print(f"True: {y_true.numpy()}")
    print(f"Loss: {loss.numpy()}")
    print(f"Gradient w.r.t. W:\n{grad_W.numpy()}")
    print(f"Gradient w.r.t. b: {grad_b.numpy()}")


def demonstrate_gradient_tape_persistence():
    """Demonstrate persistent and non-persistent GradientTape."""
    print("\n=== Gradient Tape Persistence ===")
    
    x = tf.Variable(3.0)
    
    # Non-persistent tape (default)
    with tf.GradientTape() as tape:
        y = x ** 2
        z = x ** 3
    
    # Can only compute gradient once
    grad_y = tape.gradient(y, x)
    print(f"Gradient of x^2: {grad_y.numpy()}")
    
    # This would fail because tape is already consumed:
    # grad_z = tape.gradient(z, x)  # Error!
    
    # Persistent tape allows multiple gradient computations
    with tf.GradientTape(persistent=True) as tape:
        y = x ** 2
        z = x ** 3
    
    grad_y = tape.gradient(y, x)
    grad_z = tape.gradient(z, x)
    
    print(f"Gradient of x^2: {grad_y.numpy()}")
    print(f"Gradient of x^3: {grad_z.numpy()}")
    
    # Don't forget to delete persistent tapes
    del tape


def demonstrate_watching_tensors():
    """Demonstrate watching non-Variable tensors."""
    print("\n=== Watching Tensors ===")
    
    # GradientTape automatically watches Variables
    var = tf.Variable(2.0)
    
    # But needs to explicitly watch constants/tensors
    const = tf.constant(3.0)
    
    with tf.GradientTape() as tape:
        tape.watch(const)  # Explicitly watch the constant
        
        y1 = var ** 2  # Automatically watched
        y2 = const ** 2  # Needs explicit watching
    
    grad_var = tape.gradient(y1, var)
    grad_const = tape.gradient(y2, const)
    
    print(f"Gradient w.r.t. Variable: {grad_var.numpy()}")
    print(f"Gradient w.r.t. constant: {grad_const.numpy()}")


def demonstrate_gradient_clipping():
    """Demonstrate gradient clipping techniques."""
    print("\n=== Gradient Clipping ===")
    
    # Create a model with large gradients
    x = tf.Variable([10.0, -10.0, 5.0])
    
    with tf.GradientTape() as tape:
        # Function with potentially large gradients
        y = tf.reduce_sum(tf.exp(x))  # Can produce very large gradients
    
    gradients = tape.gradient(y, x)
    print(f"Original gradients: {gradients.numpy()}")
    print(f"Gradient norms: {tf.norm(gradients).numpy()}")
    
    # Clip by norm
    clipped_by_norm = tf.clip_by_norm(gradients, clip_norm=1.0)
    print(f"Clipped by norm (1.0): {clipped_by_norm.numpy()}")
    
    # Clip by value
    clipped_by_value = tf.clip_by_value(gradients, -1.0, 1.0)
    print(f"Clipped by value [-1, 1]: {clipped_by_value.numpy()}")
    
    # Clip by global norm (useful for multiple variables)
    gradients_list = [gradients[:2], gradients[2:]]
    clipped_list, global_norm = tf.clip_by_global_norm(gradients_list, clip_norm=2.0)
    print(f"Global norm before clipping: {global_norm.numpy()}")
    print(f"Clipped by global norm: {[g.numpy() for g in clipped_list]}")


def demonstrate_stop_gradient():
    """Demonstrate stopping gradient flow."""
    print("\n=== Stopping Gradients ===")
    
    x = tf.Variable(3.0)
    
    with tf.GradientTape() as tape:
        y = x ** 2
        z = tf.stop_gradient(y) + x ** 3  # Stop gradient through y
    
    # Gradient will only flow through x^3 term
    gradient = tape.gradient(z, x)
    print(f"x = {x.numpy()}")
    print(f"y = x^2 = {y.numpy()}")
    print(f"z = stop_gradient(y) + x^3 = {z.numpy()}")
    print(f"dz/dx = {gradient.numpy()} (only from x^3 term, expected: {3 * x.numpy()**2})")


def simple_optimization_example():
    """Demonstrate a simple optimization using gradients."""
    print("\n=== Simple Optimization Example ===")
    
    # Minimize f(x) = (x - 2)^2 using gradient descent
    x = tf.Variable(0.0)
    learning_rate = 0.1
    target = 2.0
    
    print("Minimizing f(x) = (x - 2)^2 using gradient descent")
    print(f"Starting point: x = {x.numpy()}")
    
    for step in range(10):
        with tf.GradientTape() as tape:
            loss = (x - target) ** 2
        
        gradient = tape.gradient(loss, x)
        x.assign_sub(learning_rate * gradient)  # x = x - lr * gradient
        
        if step % 2 == 0:
            print(f"Step {step}: x = {x.numpy():.4f}, loss = {loss.numpy():.4f}")
    
    print(f"Final result: x = {x.numpy():.4f} (target: {target})")


def run_all_demonstrations():
    """Run all gradient tape demonstrations."""
    demonstrate_basic_gradients()
    demonstrate_gradient_of_functions()
    demonstrate_higher_order_gradients()
    demonstrate_gradient_with_neural_network()
    demonstrate_gradient_tape_persistence()
    demonstrate_watching_tensors()
    demonstrate_gradient_clipping()
    demonstrate_stop_gradient()
    simple_optimization_example()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print()
    
    run_all_demonstrations()