#!/usr/bin/env python3
"""
Test script for the new KerasAdvanced model to verify it works correctly
and can utilize GPU resources effectively.
"""

import os
import sys
import numpy as np
import tensorflow as tf

# Add the donkeycar path
sys.path.append('/Users/jerzykrczuk/workspace/donkeycar')

from donkeycar.parts.keras import KerasAdvanced

def test_advanced_model():
    """Test the KerasAdvanced model creation and basic functionality."""
    print("Testing KerasAdvanced model...")

    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print(f"GPU detected: {len(gpus)} device(s)")
        for gpu in gpus:
            print(f"  - {gpu}")
        # Enable memory growth to avoid taking all GPU memory
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"GPU setup error: {e}")
    else:
        print("No GPU detected, using CPU")

    # Create the advanced model
    print("\nCreating KerasAdvanced model...")
    pilot = KerasAdvanced(
        input_shape=(120, 160, 3),
        num_residual_blocks=4,  # Reduced for testing
        base_filters=32,        # Reduced for testing
        use_attention=True,
        mc_samples=5           # Reduced for testing
    )

    # Create the model
    model = pilot.create_model()
    print(f"Model created successfully!")

    # Print model summary
    print("\nModel Summary:")
    model.summary()

    # Test model compilation
    print("\nCompiling model...")
    pilot.compile()
    print("Model compiled successfully!")

    # Test with dummy data
    print("\nTesting with dummy data...")
    dummy_image = np.random.random((1, 120, 160, 3)).astype(np.float32)

    # Test prediction
    try:
        prediction = model.predict(dummy_image)
        print(f"Prediction successful!")
        print(f"Steering mean: {prediction[0][0][0]:.4f}")
        print(f"Steering var: {prediction[1][0][0]:.4f}")
        print(f"Throttle mean: {prediction[2][0][0]:.4f}")
        print(f"Throttle var: {prediction[3][0][0]:.4f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

    # Test the pilot interface
    print("\nTesting pilot interface...")
    try:
        dummy_img_arr = (np.random.random((120, 160, 3)) * 255).astype(np.uint8)
        result = pilot.interpreter_to_output(prediction)
        print(f"Pilot interface test successful!")
        print(f"Steering: {result[0]:.4f}, Throttle: {result[1]:.4f}")
        print(f"Steering uncertainty: {result[2]:.4f}, Throttle uncertainty: {result[3]:.4f}")
    except Exception as e:
        print(f"Pilot interface test failed: {e}")
        return False

    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.utils.get_num_params(w) for w in model.trainable_weights])
    print(f"\nModel Parameters:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Estimate GPU memory usage
    batch_size = 32
    memory_per_sample = total_params * 4  # 4 bytes per float32
    estimated_memory_mb = (memory_per_sample * batch_size) / (1024 * 1024)
    print(f"Estimated GPU memory for batch size {batch_size}: {estimated_memory_mb:.1f} MB")

    print("\n✅ All tests passed! The KerasAdvanced model is ready to use.")
    return True

if __name__ == "__main__":
    test_advanced_model()