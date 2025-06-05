import os
import numpy as np
from PIL import Image, ImageDraw
import random

def create_synthetic_iris(save_path, img_size=128, num_images=5):
    os.makedirs(save_path, exist_ok=True)
    
    for i in range(num_images):
        # Create blank grayscale image
        img = Image.new('L', (img_size, img_size), color=0)
        draw = ImageDraw.Draw(img)
        
        # Draw concentric circles to mimic iris texture
        center = (img_size // 2, img_size // 2)
        max_radius = img_size // 2 - 5
        
        for r in range(max_radius, 10, -5):
            intensity = random.randint(50, 200)
            bbox = [center[0]-r, center[1]-r, center[0]+r, center[1]+r]
            draw.ellipse(bbox, outline=intensity)
            
            # Add some random lines/rays
            for _ in range(5):
                angle = random.uniform(0, 2*np.pi)
                length = random.randint(r//2, r)
                x_end = center[0] + int(length * np.cos(angle))
                y_end = center[1] + int(length * np.sin(angle))
                draw.line([center, (x_end, y_end)], fill=intensity, width=1)
        
        # Add noise
        noise = np.random.randint(0, 30, (img_size, img_size)).astype('uint8')
        img_arr = np.array(img)
        img_arr = np.clip(img_arr + noise, 0, 255).astype('uint8')
        img = Image.fromarray(img_arr)
        
        img.save(os.path.join(save_path, f'img_{i+1}.png'))

def generate_dataset(base_dir='synthetic_iris_dataset', persons=3, images_per_person=10):
    os.makedirs(base_dir, exist_ok=True)
    for p in range(1, persons+1):
        person_folder = os.path.join(base_dir, f'person{p}')
        create_synthetic_iris(person_folder, num_images=images_per_person)
    print(f"Synthetic iris dataset generated at '{base_dir}' with {persons} persons.")

if __name__ == '__main__':
    generate_dataset()
