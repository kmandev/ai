import cv2
import os

def resize_image(img_path, target_size=500):
    # ตรวจสอบว่าไฟล์ที่ให้มามีอยู่จริงหรือไม่
    if not os.path.exists(img_path):
        print(f"Error: The file at {img_path} does not exist.")
        return None
    
    # ตรวจสอบว่าเป็นไฟล์ภาพ (.jpg, .jpeg, .png)
    valid_extensions = ['.jpg', '.jpeg', '.png']
    if not any(img_path.endswith(ext) for ext in valid_extensions):
        print(f"Error: The file at {img_path} is not a valid image.")
        return None

    # โหลดภาพ
    img = cv2.imread(img_path)
    
    # ตรวจสอบว่าโหลดภาพสำเร็จหรือไม่
    if img is None:
        print(f"Error: Unable to load image from {img_path}. Make sure the file is a valid image.")
        return None

    height, width = img.shape[:2]  # ขนาดเดิมของภาพ

    # คำนวณ aspect ratio
    if width > height:
        new_width = target_size
        new_height = int(target_size * height / width)
    else:
        new_height = target_size
        new_width = int(target_size * width / height)

    # ปรับขนาดภาพใหม่
    resized_img = cv2.resize(img, (new_width, new_height))  
    return resized_img

# ตัวอย่างการใช้งาน
def resize_images_in_folder(folder_path, output_folder, target_size=500):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        
        # ข้ามไฟล์ที่ไม่ใช่ภาพ
        if img_name.startswith('.') or not any(img_name.endswith(ext) for ext in ['.jpg', '.jpeg', '.png']):
            print(f"Skipping non-image file: {img_name}")
            continue
        
        resized_img = resize_image(img_path, target_size)
        
        # บันทึกภาพที่ปรับขนาด
        if resized_img is not None:
            output_path = os.path.join(output_folder, img_name)
            cv2.imwrite(output_path, resized_img)
            print(f"Resized image saved at {output_path}")

# ตัวอย่างการปรับขนาดภาพในโฟลเดอร์ที่เก็บภาพมือสะอาดและมือสกปรก
resize_images_in_folder('dataset/clean', 'dataset/clean_resized', 500)
resize_images_in_folder('dataset/dirty', 'dataset/dirty_resized', 500)
