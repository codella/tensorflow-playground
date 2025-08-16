"""
Data Augmentation Techniques

This module demonstrates various data augmentation techniques for images and text.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def demonstrate_image_augmentation():
    """Demonstrate various image augmentation techniques."""
    print("=== Image Augmentation Techniques ===")
    
    # Create a dummy image
    image = tf.random.uniform([224, 224, 3], dtype=tf.float32)
    
    # Basic augmentations
    def flip_augmentation(image):
        """Flip augmentations."""
        flipped_lr = tf.image.flip_left_right(image)
        flipped_ud = tf.image.flip_up_down(image)
        return flipped_lr, flipped_ud
    
    def rotation_augmentation(image):
        """Rotation augmentations."""
        # Random rotation
        angle = tf.random.uniform([], -0.2, 0.2)  # radians
        rotated = tf.contrib.image.rotate(image, angle) if hasattr(tf.contrib, 'image') else image
        
        # 90-degree rotations
        rot90 = tf.image.rot90(image, k=1)
        rot180 = tf.image.rot90(image, k=2)
        rot270 = tf.image.rot90(image, k=3)
        
        return rotated, rot90, rot180, rot270
    
    def brightness_contrast_augmentation(image):
        """Brightness and contrast augmentations."""
        # Random brightness
        bright = tf.image.random_brightness(image, max_delta=0.2)
        
        # Random contrast
        contrast = tf.image.random_contrast(image, lower=0.8, upper=1.2)
        
        # Combined
        combined = tf.image.random_brightness(image, max_delta=0.1)
        combined = tf.image.random_contrast(combined, lower=0.9, upper=1.1)
        
        return bright, contrast, combined
    
    def color_augmentation(image):
        """Color-based augmentations."""
        # Random hue
        hue = tf.image.random_hue(image, max_delta=0.1)
        
        # Random saturation
        saturation = tf.image.random_saturation(image, lower=0.8, upper=1.2)
        
        # RGB to grayscale
        grayscale = tf.image.rgb_to_grayscale(image)
        grayscale = tf.image.grayscale_to_rgb(grayscale)  # Convert back to 3 channels
        
        return hue, saturation, grayscale
    
    def crop_resize_augmentation(image):
        """Crop and resize augmentations."""
        # Random crop
        cropped = tf.image.random_crop(image, size=[200, 200, 3])
        
        # Central crop
        central_crop = tf.image.central_crop(image, central_fraction=0.8)
        
        # Resize with different methods
        resized_bilinear = tf.image.resize(image, [256, 256], method='bilinear')
        resized_nearest = tf.image.resize(image, [256, 256], method='nearest')
        
        # Crop and resize
        crop_and_resize = tf.image.resize(cropped, [224, 224])
        
        return cropped, central_crop, resized_bilinear, resized_nearest, crop_and_resize
    
    def noise_augmentation(image):
        """Add noise to images."""
        # Gaussian noise
        noise = tf.random.normal(tf.shape(image), mean=0.0, stddev=0.1)
        noisy = tf.clip_by_value(image + noise, 0.0, 1.0)
        
        # Salt and pepper noise simulation
        mask = tf.random.uniform(tf.shape(image))
        salt_pepper = tf.where(mask < 0.05, 1.0, image)  # Salt
        salt_pepper = tf.where(mask > 0.95, 0.0, salt_pepper)  # Pepper
        
        return noisy, salt_pepper
    
    print("Image augmentation functions created successfully")
    
    # Test basic augmentations
    flipped_lr, flipped_ud = flip_augmentation(image)
    print(f"Original shape: {image.shape}")
    print(f"Flipped LR shape: {flipped_lr.shape}")
    print(f"Flipped UD shape: {flipped_ud.shape}")


def create_augmentation_layer():
    """Create a Keras preprocessing layer for augmentation."""
    print("\n=== Keras Preprocessing Layers ===")
    
    # Create individual preprocessing layers
    augmentation_layers = [
        tf.keras.layers.RandomFlip("horizontal_and_vertical"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
        tf.keras.layers.RandomBrightness(0.1),
    ]
    
    # Combine into a sequential model
    data_augmentation = tf.keras.Sequential(augmentation_layers, name="data_augmentation")
    
    # Test with dummy data
    dummy_image = tf.random.uniform([1, 224, 224, 3])
    augmented = data_augmentation(dummy_image, training=True)
    
    print(f"Original image shape: {dummy_image.shape}")
    print(f"Augmented image shape: {augmented.shape}")
    print("Keras preprocessing layers created successfully")
    
    return data_augmentation


def demonstrate_advanced_augmentation():
    """Demonstrate advanced augmentation techniques."""
    print("\n=== Advanced Augmentation Techniques ===")
    
    # Mixup augmentation
    def mixup_augmentation(images, labels, alpha=0.2):
        """Implement Mixup augmentation."""
        batch_size = tf.shape(images)[0]
        
        # Sample lambda from Beta distribution
        lambda_val = tf.random.gamma([batch_size, 1, 1, 1], alpha, alpha)
        lambda_val = tf.minimum(lambda_val, 1.0 - lambda_val)
        
        # Shuffle indices
        indices = tf.random.shuffle(tf.range(batch_size))
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        # Mix images and labels
        mixed_images = lambda_val * images + (1 - lambda_val) * shuffled_images
        mixed_labels = tf.squeeze(lambda_val[:, 0, 0, :], axis=-1)[:, None] * labels + \
                      (1 - tf.squeeze(lambda_val[:, 0, 0, :], axis=-1)[:, None]) * shuffled_labels
        
        return mixed_images, mixed_labels
    
    # CutMix augmentation
    def cutmix_augmentation(images, labels, alpha=1.0):
        """Implement CutMix augmentation."""
        batch_size = tf.shape(images)[0]
        height, width = tf.shape(images)[1], tf.shape(images)[2]
        
        # Sample lambda
        lambda_val = tf.random.uniform([batch_size], 0, 1)
        
        # Generate random box
        cut_ratio = tf.sqrt(1.0 - lambda_val)
        cut_w = tf.cast(cut_ratio * tf.cast(width, tf.float32), tf.int32)
        cut_h = tf.cast(cut_ratio * tf.cast(height, tf.float32), tf.int32)
        
        cx = tf.random.uniform([batch_size], 0, width, dtype=tf.int32)
        cy = tf.random.uniform([batch_size], 0, height, dtype=tf.int32)
        
        bbx1 = tf.clip_by_value(cx - cut_w // 2, 0, width)
        bby1 = tf.clip_by_value(cy - cut_h // 2, 0, height)
        bbx2 = tf.clip_by_value(cx + cut_w // 2, 0, width)
        bby2 = tf.clip_by_value(cy + cut_h // 2, 0, height)
        
        # This is a simplified version - full implementation would require more complex tensor ops
        shuffled_images = tf.gather(images, tf.random.shuffle(tf.range(batch_size)))
        
        # For demonstration, we'll return the shuffled images (actual CutMix requires patch replacement)
        return shuffled_images, labels
    
    # AutoAugment policies
    def autoaugment_policy():
        """Create an AutoAugment-style policy."""
        augment_layers = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.15),
            tf.keras.layers.RandomTranslation(0.1, 0.1),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.15),
            tf.keras.layers.RandomBrightness(0.15),
        ])
        return augment_layers
    
    # Test advanced augmentations
    dummy_images = tf.random.uniform([4, 32, 32, 3])
    dummy_labels = tf.one_hot(tf.random.uniform([4], 0, 10, dtype=tf.int32), 10)
    
    # Test Mixup
    mixed_images, mixed_labels = mixup_augmentation(dummy_images, dummy_labels)
    print(f"Mixup - Original shape: {dummy_images.shape}, Mixed shape: {mixed_images.shape}")
    
    # Test CutMix
    cutmix_images, cutmix_labels = cutmix_augmentation(dummy_images, dummy_labels)
    print(f"CutMix - Original shape: {dummy_images.shape}, CutMix shape: {cutmix_images.shape}")
    
    # Test AutoAugment
    autoaugment = autoaugment_policy()
    augmented_images = autoaugment(dummy_images)
    print(f"AutoAugment - Original shape: {dummy_images.shape}, Augmented shape: {augmented_images.shape}")


def demonstrate_text_augmentation():
    """Demonstrate text augmentation techniques."""
    print("\n=== Text Augmentation Techniques ===")
    
    # Sample texts
    texts = [
        "This is a great movie with excellent acting",
        "The film was terrible and boring",
        "Amazing cinematography and compelling story",
        "Worst movie I have ever seen"
    ]
    
    # Synonym replacement (conceptual - would need word embeddings in practice)
    def synonym_replacement(text, n=1):
        """Replace n words with synonyms."""
        # This is a simplified version - real implementation would use word embeddings
        synonyms = {
            "great": "excellent",
            "terrible": "awful", 
            "amazing": "fantastic",
            "worst": "terrible"
        }
        
        words = text.split()
        for i, word in enumerate(words):
            if word in synonyms and np.random.random() < 0.3:
                words[i] = synonyms[word]
        
        return " ".join(words)
    
    # Random insertion
    def random_insertion(text, n=1):
        """Randomly insert n words."""
        words = text.split()
        for _ in range(n):
            new_word = np.random.choice(["very", "really", "quite", "extremely"])
            random_idx = np.random.randint(0, len(words) + 1)
            words.insert(random_idx, new_word)
        return " ".join(words)
    
    # Random swap
    def random_swap(text, n=1):
        """Randomly swap n pairs of words."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for _ in range(n):
            idx1, idx2 = np.random.choice(len(words), 2, replace=False)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return " ".join(words)
    
    # Random deletion
    def random_deletion(text, p=0.1):
        """Randomly delete words with probability p."""
        words = text.split()
        if len(words) == 1:
            return text
        
        new_words = [word for word in words if np.random.random() > p]
        if len(new_words) == 0:
            return words[0]  # Return at least one word
        
        return " ".join(new_words)
    
    # Test text augmentations
    print("Original texts and augmentations:")
    for i, text in enumerate(texts):
        print(f"\n{i+1}. Original: {text}")
        print(f"   Synonym replacement: {synonym_replacement(text)}")
        print(f"   Random insertion: {random_insertion(text)}")
        print(f"   Random swap: {random_swap(text)}")
        print(f"   Random deletion: {random_deletion(text)}")


def create_augmentation_pipeline():
    """Create a complete augmentation pipeline for training."""
    print("\n=== Complete Augmentation Pipeline ===")
    
    # Image augmentation function
    @tf.function
    def augment_image(image, label):
        # Random flip
        image = tf.image.random_flip_left_right(image)
        
        # Random rotation (using small angle)
        image = tf.image.rot90(image, k=tf.random.uniform([], 0, 4, dtype=tf.int32))
        
        # Random brightness and contrast
        image = tf.image.random_brightness(image, max_delta=0.1)
        image = tf.image.random_contrast(image, lower=0.9, upper=1.1)
        
        # Random crop and resize
        image = tf.image.random_crop(image, [28, 28, tf.shape(image)[-1]])
        image = tf.image.resize(image, [32, 32])
        
        # Ensure values are in [0, 1]
        image = tf.clip_by_value(image, 0.0, 1.0)
        
        return image, label
    
    # Create dummy dataset
    def create_dummy_dataset():
        images = tf.random.uniform([100, 32, 32, 3])
        labels = tf.random.uniform([100], 0, 10, dtype=tf.int32)
        return tf.data.Dataset.from_tensor_slices((images, labels))
    
    # Build pipeline
    dataset = create_dummy_dataset()
    
    # Apply augmentation
    augmented_dataset = (dataset
                        .map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
                        .batch(16)
                        .prefetch(tf.data.AUTOTUNE))
    
    print("Augmentation pipeline created")
    
    # Test pipeline
    for images, labels in augmented_dataset.take(1):
        print(f"Batch shape: {images.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Image value range: [{tf.reduce_min(images):.3f}, {tf.reduce_max(images):.3f}]")
    
    return augmented_dataset


def demonstrate_conditional_augmentation():
    """Demonstrate conditional augmentation based on labels or other factors."""
    print("\n=== Conditional Augmentation ===")
    
    @tf.function
    def conditional_augment(image, label):
        # Different augmentation strategies based on label
        
        # For class 0: light augmentation
        def light_augment(img):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_brightness(img, max_delta=0.05)
            return img
        
        # For class 1: heavy augmentation
        def heavy_augment(img):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.2)
            img = tf.image.random_contrast(img, lower=0.7, upper=1.3)
            return img
        
        # Apply different augmentation based on label
        image = tf.cond(
            tf.equal(label % 2, 0),  # Even labels get light augmentation
            lambda: light_augment(image),
            lambda: heavy_augment(image)
        )
        
        return image, label
    
    # Test conditional augmentation
    dummy_images = tf.random.uniform([5, 32, 32, 3])
    dummy_labels = tf.range(5)  # Labels 0, 1, 2, 3, 4
    
    dataset = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels))
    augmented_dataset = dataset.map(conditional_augment)
    
    print("Conditional augmentation applied based on label parity")
    for image, label in augmented_dataset:
        augmentation_type = "light" if label % 2 == 0 else "heavy"
        print(f"Label {label.numpy()}: {augmentation_type} augmentation applied")


def run_all_demonstrations():
    """Run all data augmentation demonstrations."""
    demonstrate_image_augmentation()
    create_augmentation_layer()
    demonstrate_advanced_augmentation()
    demonstrate_text_augmentation()
    create_augmentation_pipeline()
    demonstrate_conditional_augmentation()


if __name__ == "__main__":
    print("TensorFlow version:", tf.__version__)
    print()
    
    run_all_demonstrations()