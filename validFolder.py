import os

# ตรวจสอบไฟล์ในโฟลเดอร์
clean_images = os.listdir('dataset/clean_resized')
dirty_images = os.listdir('dataset/dirty_resized')

print(f"Clean images: {clean_images}")
print(f"Dirty images: {dirty_images}")

# ตรวจสอบว่าไฟล์ในโฟลเดอร์เป็นภาพ
valid_extensions = ['.jpg', '.jpeg', '.png']
clean_images = [img for img in clean_images if any(img.endswith(ext) for ext in valid_extensions)]
dirty_images = [img for img in dirty_images if any(img.endswith(ext) for ext in valid_extensions)]

print(f"Valid clean images: {clean_images}")
print(f"Valid dirty images: {dirty_images}")
