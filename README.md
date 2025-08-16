# TensorFlow Capabilities Tutorial

A comprehensive project for testing and exploring TensorFlow's fundamental capabilities using the latest APIs and best practices.

## Project Overview

This project provides a systematic exploration of TensorFlow's core features, from basic tensor operations to advanced model architectures and deployment strategies. Each module is designed to be educational, practical, and based on the latest TensorFlow 2.15+ APIs.

## Project Structure

```
tensorflow-tutorial/
├── src/
│   ├── 01_fundamentals/          # Core TensorFlow concepts
│   │   ├── tensor_basics.py      # Tensor operations, shapes, dtypes
│   │   ├── eager_execution.py    # Eager vs graph execution
│   │   └── gradient_tape.py      # Automatic differentiation
│   ├── 02_keras_models/          # Keras API demonstrations  
│   │   ├── sequential_model.py   # Sequential API
│   │   ├── functional_api.py     # Functional API
│   │   ├── model_subclassing.py  # Model subclassing
│   │   └── custom_layers.py      # Custom layer implementation
│   ├── 03_data_pipeline/         # tf.data API
│   │   ├── dataset_basics.py     # Dataset creation and manipulation
│   │   ├── data_augmentation.py  # Image/text augmentation
│   │   └── prefetch_cache.py     # Performance optimization
│   ├── 04_training/              # Training techniques (planned)
│   ├── 05_pretrained_models/     # Transfer learning (planned)
│   ├── 06_model_optimization/    # Performance optimization (planned)
│   ├── 07_deployment/            # Model deployment (planned)
│   └── utils/                    # Helper utilities (planned)
├── examples/                     # Complete example applications (planned)
├── notebooks/                    # Jupyter notebooks (planned)
├── tests/                        # Unit tests (planned)
├── requirements.txt              # Dependencies
├── setup.py                      # Package setup
└── .gitignore                    # Git ignore rules
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd tensorflow-tutorial
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv tf_env
   source tf_env/bin/activate  # On Windows: tf_env\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   Or install in development mode:
   ```bash
   pip install -e .
   ```

### Dependencies

- TensorFlow ≥ 2.15.0
- TensorFlow Hub ≥ 0.15.0
- TensorFlow Datasets ≥ 4.9.0
- NumPy ≥ 1.24.0
- Matplotlib ≥ 3.7.0
- Additional utilities (see `requirements.txt`)

## How to Run

### Individual Modules

Each module can be run independently to explore specific TensorFlow capabilities:

#### 1. Tensor Fundamentals
```bash
# Basic tensor operations
python src/01_fundamentals/tensor_basics.py

# Eager execution vs tf.function
python src/01_fundamentals/eager_execution.py

# Automatic differentiation
python src/01_fundamentals/gradient_tape.py
```

#### 2. Keras Models
```bash
# Sequential API examples
python src/02_keras_models/sequential_model.py

# Functional API examples
python src/02_keras_models/functional_api.py

# Model subclassing examples
python src/02_keras_models/model_subclassing.py

# Custom layers
python src/02_keras_models/custom_layers.py
```

#### 3. Data Pipeline
```bash
# tf.data basics
python src/03_data_pipeline/dataset_basics.py

# Data augmentation techniques
python src/03_data_pipeline/data_augmentation.py

# Performance optimization
python src/03_data_pipeline/prefetch_cache.py
```

### Running All Demonstrations

To run all available demonstrations in sequence:

```bash
# Run from project root
python -c "
import sys
sys.path.append('src')

# Fundamentals
from src.01_fundamentals import tensor_basics, eager_execution, gradient_tape
from src.02_keras_models import sequential_model, functional_api, model_subclassing, custom_layers
from src.03_data_pipeline import dataset_basics, data_augmentation, prefetch_cache

print('=== TENSOR FUNDAMENTALS ===')
tensor_basics.run_all_demonstrations()
eager_execution.run_all_demonstrations()
gradient_tape.run_all_demonstrations()

print('\n=== KERAS MODELS ===')
sequential_model.run_all_demonstrations()
functional_api.run_all_demonstrations()
model_subclassing.run_all_demonstrations()
custom_layers.run_all_demonstrations()

print('\n=== DATA PIPELINE ===')
dataset_basics.run_all_demonstrations()
data_augmentation.run_all_demonstrations()
prefetch_cache.run_all_demonstrations()
"
```

## Module Descriptions

### 01_fundamentals/
- **tensor_basics.py**: Learn tensor creation, properties, operations, indexing, and Variables
- **eager_execution.py**: Compare eager execution vs graph execution, understand tf.function
- **gradient_tape.py**: Master automatic differentiation and custom training loops

### 02_keras_models/
- **sequential_model.py**: Build models using the Sequential API for common architectures
- **functional_api.py**: Create complex models with multiple inputs/outputs using Functional API  
- **model_subclassing.py**: Implement custom models with full control using subclassing
- **custom_layers.py**: Create custom layers including attention mechanisms and transformers

### 03_data_pipeline/
- **dataset_basics.py**: Master tf.data API for efficient data loading and preprocessing
- **data_augmentation.py**: Implement image and text augmentation for better model performance
- **prefetch_cache.py**: Optimize pipeline performance with caching and prefetching

## Learning Path

### Beginner Path
1. Start with `01_fundamentals/tensor_basics.py` - Learn core tensor operations
2. Explore `02_keras_models/sequential_model.py` - Build your first neural networks
3. Try `03_data_pipeline/dataset_basics.py` - Understand data loading

### Intermediate Path
1. Dive into `01_fundamentals/gradient_tape.py` - Master automatic differentiation
2. Explore `02_keras_models/functional_api.py` - Build complex architectures
3. Optimize with `03_data_pipeline/prefetch_cache.py` - Learn performance optimization

### Advanced Path
1. Master `02_keras_models/model_subclassing.py` - Full control over model behavior
2. Create custom components with `02_keras_models/custom_layers.py`
3. Implement advanced augmentation with `03_data_pipeline/data_augmentation.py`

## Development

### Running Tests
```bash
python -m pytest tests/
```

### Code Style
```bash
# Format code
black src/

# Check style
flake8 src/
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add your improvements
4. Ensure tests pass
5. Submit a pull request

## Additional Resources

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras API Reference](https://keras.io/api/)
- [tf.data Performance Guide](https://www.tensorflow.org/guide/data_performance)

## Troubleshooting

### Common Issues

1. **GPU not detected:**
   ```bash
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

2. **Memory issues:**
   - Reduce batch sizes in examples
   - Use `tf.config.experimental.set_memory_growth()` for GPU

3. **Import errors:**
   - Ensure you're in the project root directory
   - Check that all dependencies are installed

### System Requirements

- **CPU**: Any modern x86_64 processor
- **Memory**: 8GB RAM minimum (16GB recommended)
- **GPU**: Optional, but recommended for larger models
- **Storage**: 2GB free space

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## What's Next?

This project is actively being developed. Upcoming modules include:

- **04_training/**: Advanced training techniques, callbacks, distributed training
- **05_pretrained_models/**: Transfer learning and fine-tuning
- **06_model_optimization/**: Mixed precision, XLA compilation, quantization
- **07_deployment/**: SavedModel format, TensorFlow Lite, TensorFlow Serving
- **examples/**: Complete end-to-end applications
- **notebooks/**: Interactive Jupyter notebooks

---

Happy learning with TensorFlow!
