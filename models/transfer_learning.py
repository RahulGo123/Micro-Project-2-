import tensorflow as tf
import numpy as np
import os

BATCH_SIZE = 32
IMG_SIZE = (224, 224)
DATA_DIR = os.path.join("data", "raw")

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="training",
    seed=235,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    validation_split=0.2,
    subset="validation",
    seed=235,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
)

class_names = train_ds.class_names

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(),
        tf.keras.layers.RandomRotation(0.2),
    ]
)

base_model = tf.keras.applications.ResNet50(
    input_shape=(224, 224, 3), include_top=False, weights="imagenet"
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

x = tf.keras.applications.resnet50.preprocess_input(x)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(train_ds, epochs=5, validation_data=val_ds)

model.save("models/satellite_resnet.keras")


# Grab one batch of 32 images
image_batch, label_batch = next(iter(val_ds))
image = image_batch[0]
label = label_batch[0]

# Predict
# Note: We must expand dims to (1, 224, 224, 3) for the model
prediction = model.predict(tf.expand_dims(image, axis=0))
predicted_class_index = np.argmax(prediction)
predicted_class_name = class_names[predicted_class_index]
confidence = np.max(prediction)

true_class_name = class_names[label]

print(f"--------------------------------")
print(f"TRUE LABEL:      {true_class_name}")
print(f"PREDICTED LABEL: {predicted_class_name}")
print(f"CONFIDENCE:      {confidence:.2%}")
print(f"--------------------------------")

if true_class_name == predicted_class_name:
    print("🎉 SUCCESS: The model identified it correctly!")
else:
    print("❌ FAIL: The model missed it.")
