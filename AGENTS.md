# AGENTS.md - Coding Guidelines for Donkeycar

This document provides essential information for agentic coding assistants working with the Donkeycar codebase.

## Build Commands

### Installation
```bash
# Install donkeycar with all dependencies
pip install -e .[pc,dev]

# For Raspberry Pi
pip install -e .[pi,dev]

# For NVIDIA Jetson Nano
pip install -e .[nano,dev]

# For macOS with Apple Silicon
pip install -e .[macos,dev]
```

### Building Packages
```bash
# Create source distribution
python setup.py sdist

# Or using Makefile
make package
```

## Linting Commands

### Code Formatting
```bash
# Format code with Black (configured in .github/linters/.python-black)
black .

# Check formatting without making changes
black --check .
```

### General Linting
```bash
# Run super-linter (as in CI)
# Uses GitHub Actions workflow in .github/workflows/superlinter.yml
```

## Test Commands

### Running All Tests
```bash
# Run all tests
pytest

# Or using Makefile
make tests
```

### Running Specific Tests
```bash
# Run a specific test file
pytest donkeycar/tests/test_vehicle.py

# Run a specific test function
pytest donkeycar/tests/test_vehicle.py::test_create_vehicle

# Run tests with coverage
pytest --cov=donkeycar

# Run tests in parallel for faster execution
pytest -n auto
```

### Test Structure
Tests are located in `donkeycar/tests/` and use pytest framework:
- Unit tests for individual components
- Integration tests for vehicle configurations
- Mock hardware when needed for testing

## Code Style Guidelines

### Imports
1. Standard library imports first (sorted alphabetically)
2. Third-party imports second (sorted alphabetically)
3. Local application imports last (sorted alphabetically)
4. Use explicit imports rather than wildcards
5. Group imports by type with blank lines between groups

```python
# Good
import os
import sys
from typing import List, Optional
import numpy as np
import tensorflow as tf
from donkeycar.parts.camera import Camera
from donkeycar.vehicle import Vehicle
```

### Formatting
1. Follow PEP 8 Python style guide
2. Line length: 80 characters (configured in Black settings)
3. Use 4 spaces for indentation (no tabs)
4. Use Black for automatic formatting:
   - `skip-string-normalization = true` (preserve single quotes)
   - `include_trailing_comma = true`
   - `multi_line_output = 3`

### Type Hints
1. Use type hints for function parameters and return values
2. Import typing module for complex types
3. Use Optional[T] for parameters that can be None
4. Use Union[T1, T2] for multiple possible types

```python
from typing import Optional, List, Tuple

def process_images(images: List[np.ndarray], 
                  threshold: float = 0.5) -> Optional[Tuple[np.ndarray, float]]:
    # Implementation
    pass
```

### Naming Conventions
1. Classes: PascalCase (e.g., `Vehicle`, `CameraPart`)
2. Functions and variables: snake_case (e.g., `get_image`, `max_speed`)
3. Constants: UPPER_SNAKE_CASE (e.g., `MAX_THROTTLE`, `IMAGE_WIDTH`)
4. Private members: prefixed with underscore (e.g., `_private_method`)
5. Module names: short, lowercase, no underscores if possible

### Error Handling
1. Use specific exception types rather than generic `Exception`
2. Include meaningful error messages
3. Log errors appropriately using the logging module
4. Handle exceptions close to where they occur
5. Use context managers (`with` statements) for resource management

```python
import logging

logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        raise
```

### Documentation
1. Use docstrings for all public classes and functions
2. Follow Google Python Style Guide for docstrings
3. Include type information in docstrings
4. Document parameters, return values, and exceptions

```python
def calculate_steering_angle(image: np.ndarray, 
                           model: tf.keras.Model) -> float:
    """Calculate steering angle from image using ML model.
    
    Args:
        image: Input image as numpy array with shape (height, width, channels)
        model: Trained TensorFlow model for steering prediction
        
    Returns:
        float: Predicted steering angle in range [-1, 1]
        
    Raises:
        ValueError: If image dimensions don't match model input
    """
    # Implementation
    pass
```

## Architecture Patterns

### Vehicle/Part System
Donkeycar uses a modular architecture where:
1. `Vehicle` is the main container that manages all parts
2. `Part` is the base interface for components
3. Parts communicate through a shared `Memory` system
4. Data flows through parts in a pipeline

### Part Implementation
All parts should follow this pattern:
```python
class MyPart:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2
        
    def run(self, input1, input2):
        # Process inputs and return outputs
        return output1, output2
        
    def run_threaded(self, input1, input2):
        # Optional threaded implementation
        pass
        
    def shutdown(self):
        # Optional cleanup
        pass
```

### Configuration System
1. Use `config.py` for default settings
2. Use `myconfig.py` for user overrides
3. All config variables should be UPPER_SNAKE_CASE
4. Document config variables clearly

## Testing Guidelines

### Test Structure
1. Use pytest framework
2. Follow naming convention: `test_*.py` for test files
3. Use fixtures for setup/teardown
4. Mock external dependencies
5. Test both positive and negative cases

### Example Test
```python
import pytest
from donkeycar.parts.camera import Camera

def test_camera_initialization():
    """Test camera initializes with correct parameters."""
    camera = Camera(width=160, height=120)
    assert camera.width == 160
    assert camera.height == 120

def test_camera_run():
    """Test camera produces image output."""
    camera = Camera()
    image = camera.run()
    assert image is not None
    assert isinstance(image, np.ndarray)
```

## Performance Considerations

1. Use threading for I/O bound operations (camera, network)
2. Profile code before optimizing
3. Consider using NumPy vectorized operations
4. Minimize memory allocations in loops
5. Use appropriate data structures (e.g., deque for circular buffers)

## Hardware Integration Guidelines

1. Abstract hardware dependencies behind Part interfaces
2. Provide software simulation where possible
3. Handle hardware failures gracefully
4. Document hardware requirements clearly
5. Use conditional imports for platform-specific code

This guide is specifically for agentic coding assistants and may be updated as the project evolves.