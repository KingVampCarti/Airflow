import sys
import pathlib
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os
import mlflow

print(tf.test.gpu_device_name())

data_path_args = sys.argv[1]
run_date = sys.argv[2]
MLFLOW_SERVER = sys.argv[3]
data_dir = pathlib.Path("{}/train/".format(data_path_args))
batch_size = 32
img_height = 224
img_width = 224
img_size = (224, 224)

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width))

train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_dataset.class_names

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal", input_shape=(img_height, img_width,3)),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.1),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.1),
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
img_shape = img_size + (3,)

base_model = tf.keras.applications.MobileNetV2(
  input_shape=img_shape,
  include_top=False,
  weights='imagenet'
)

image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)

base_model.trainable = False

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0001
model.compile(
  optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
  metrics=['accuracy']
)

initial_epochs = 10
loss0, accuracy0 = model.evaluate(validation_dataset)
print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

mlflow.set_tracking_uri(f"http://{MLFLOW_SERVER}:5000")
mlflow.keras.autolog(registered_model_name=f"xray_classifier_model")
if mlflow.get_experiment_by_name(f"run_{run_date}") == None:
  mlflow.create_experiment(f"run_{run_date}")
mlflow.set_experiment(f"run_{run_date}")
with mlflow.start_run() as run:
  history = model.fit(
    train_dataset,
    epochs=initial_epochs,
    validation_data=validation_dataset
  )

  acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  base_model.trainable = True
  fine_tune_at = 100
  for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

  model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
    metrics=['accuracy']
  )

  fine_tune_epochs = 15
  total_epochs =  initial_epochs + fine_tune_epochs

  history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=validation_dataset
  )

  acc += history_fine.history['accuracy']
  val_acc += history_fine.history['val_accuracy']
  loss += history_fine.history['loss']
  val_loss += history_fine.history['val_loss']

loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)

os.makedirs('{}/models/{}/'.format(data_path_args,run_date))
model.save("{}/models/{}/xray_classifier_model.h5".format(data_path_args,run_date))
