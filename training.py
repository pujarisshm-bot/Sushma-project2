import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- STEP 1: DATA PROCESSING ---
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

test_set = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(224, 224),
    batch_size=32,
    class_mode='sparse'
)

# --- STEP 2: MODEL BUILDING (VGG16) ---
vgg = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze VGG16 layers
for layer in vgg.layers:
    layer.trainable = False

model = Sequential([
    vgg,
    Flatten(),
    Dropout(0.5),
    Dense(6, activation='softmax') # 5 categories
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Data processing ready and Model architecture built!")
# --- STEP 3: START TRAINING ---
print("Starting training... please wait.")
model.fit(
    training_set,
    validation_data=test_set,
    epochs=10
)

# --- STEP 4: SAVE THE MODEL ---
model.save('model.h5')
print("Training complete! Model saved as model.h5")