import os
import tensorflow as tf

from src.data_loader import (
    download_dataset,
    create_generators
)

from src.classifier import (
    build_model,
    train_model,
    evaluate_model
)

from src.visualization import (
    plot_loss,
    plot_accuracy
)

from src.captioning import (
    generate_text
)

dataset_path = download_dataset()

train_dir = os.path.join(
    dataset_path,
    "train"
)

valid_dir = os.path.join(
    dataset_path,
    "valid"
)

test_dir = os.path.join(
    dataset_path,
    "test"
)

train_gen, valid_gen, test_gen = create_generators(
    train_dir,
    valid_dir,
    test_dir
)

model = build_model()

history = train_model(
    model,
    train_gen,
    valid_gen
)

plot_loss(history)

plot_accuracy(history)

evaluate_model(
    model,
    test_gen
)

image_path = tf.constant(
    "aircraft_damage_dataset_v1/test/dent/sample.jpg"
)

caption = generate_text(
    image_path,
    tf.constant("caption")
)

summary = generate_text(
    image_path,
    tf.constant("summary")
)

print(
    caption.numpy().decode("utf-8")
)

print(
    summary.numpy().decode("utf-8")
)
