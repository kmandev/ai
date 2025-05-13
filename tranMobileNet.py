import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.legacy import Adam

# สร้าง ImageDataGenerator สำหรับข้อมูลฝึกและทดสอบ
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True, rotation_range=20)
val_datagen = ImageDataGenerator(rescale=1./255)

# โหลดข้อมูลจากโฟลเดอร์
train_data = train_datagen.flow_from_directory(
    'dataset',  # เส้นทางของภาพมือสะอาด
    target_size=(224, 500),  # ขนาดภาพที่ต้องการ
    batch_size=32,           # ขนาด batch
    class_mode='binary',      # สำหรับการจำแนก 2 คลาส
    subset='training'
)

val_data = val_datagen.flow_from_directory(
    'dataset',  # เส้นทางของภาพมือสกปรก
    target_size=(224, 500),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# ตรวจสอบข้อมูลที่โหลดได้
print(f"Found {train_data.samples} training images")
print(f"Found {val_data.samples} validation images")

# โหลดโมเดลที่ฝึกมาแล้ว (MobileNetV2)
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze layers ของโมเดลที่ฝึกมาแล้ว
base_model.trainable = False

# สร้างโมเดลใหม่จากโมเดลที่ฝึกมาแล้ว
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # ใช้ sigmoid สำหรับ binary classification
])

# คอมไพล์โมเดล
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# ฝึกโมเดล
model.fit(train_data, epochs=10, validation_data=val_data)

# บันทึกโมเดลที่ฝึกเสร็จแล้ว
model.save('hand_scan_model.h5')

# โหลดโมเดลที่ฝึกเสร็จแล้ว
model = tf.keras.models.load_model('hand_scan_model.h5')

# ---------- CONVERT TO .tflite AND OPTIMIZE ----------
# แปลงโมเดลเป็น .tflite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ทำการ Optimize โมเดล
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # ใช้การ Quantization

# แปลงโมเดล
tflite_model = converter.convert()

# บันทึกโมเดลที่แปลงแล้วเป็น .tflite
with open('hand_clean_model_optimized.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ โมเดล .tflite ได้รับการแปลงและบันทึกเรียบร้อยแล้ว")
