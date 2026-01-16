import tensorflow as tf
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

preprocess_func = tf.keras.applications.resnet50.preprocess_input


def preprocess(image, label):
    return preprocess_func(image), label


train_ds = train_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.map(preprocess).prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.load_model("models/satellite_resnet.keras")

base_model = None
for layer in model.layers:
    if "resnet" in layer.name:
        base_model = layer
        break

if base_model:

    base_model.trainable = True

    for layer in base_model.layers[:-10]:
        layer.trainable = False

else:
    exit()

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5)

model.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)


history = model.fit(train_ds, epochs=5, validation_data=val_ds)

model.save("models/satellite_resnet_finetuned.keras")
