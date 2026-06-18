import os
import csv
import random

# --- Configuration ---
from backend.app.config import BASE_DATASET_DIR

IMAGE_DIR = BASE_DATASET_DIR
OUTPUT_CSV = "mango_metadata.csv"

def generate_mock_data():
    classes = ['unripe', 'partially_ripe', 'ripe']
    data = []
    
    # Check if directory exists
    if not os.path.exists(IMAGE_DIR):
        print(f"Error: Cannot find directory {IMAGE_DIR}")
        return

    print(f"Scanning {IMAGE_DIR} for images...")
    
    for class_name in classes:
        class_dir = os.path.join(IMAGE_DIR, class_name)
        if not os.path.exists(class_dir):
            print(f"Skipping {class_name}: Folder not found.")
            continue
            
        for filename in os.listdir(class_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                # Create a relative path like "ripe/327(4).jpg"
                rel_path = f"{class_name}/{filename}"
                
                # Generate realistic random environmental data
                temp = round(random.uniform(25.0, 35.0), 1)
                humidity = random.randint(50, 85)
                
                # Assign logical harvest days based on current ripeness
                if class_name == 'ripe':
                    days = random.randint(1, 3)
                elif class_name == 'partially_ripe':
                    days = random.randint(4, 9)
                else: # unripe
                    days = random.randint(10, 18)
                    
                data.append([rel_path, temp, humidity, days])

    # Write to CSV
    with open(OUTPUT_CSV, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'temperature', 'humidity', 'days_to_harvest'])
        writer.writerows(data)
        
    print(f"Success! Generated mock metadata for {len(data)} images.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    generate_mock_data()