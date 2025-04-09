import gradio as gr
import torch
from torchvision import models, transforms
from PIL import Image as PILImage
import torch.nn as nn
import torch.nn.functional as F
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from threading import Thread
import os
import time

# Define the model architecture
model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)  # Use pretrained weights
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 3)  # Adjust based on the number of classes

# Load the saved model weights
model.load_state_dict(torch.load('/home/aarabambi/Meat Project/meat2/model_weights.pth'))
model.eval()  # Set the model to evaluation mode

# Define the image transformation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

results = []  # Global list to store results
processed_files = set()  # Set to keep track of processed files

def classify_image(file_path):
    if file_path in processed_files:
        return  # Skip processing if the file has already been processed
    processed_files.add(file_path)  # Mark the file as processed

    # Attempt to open the image with retries
    max_attempts = 5
    for attempt in range(max_attempts):
        try:
            image = PILImage.open(file_path).convert('RGB')
            display_image = image.copy()  # Copy image for display
            break
        except IOError as e:
            if attempt < max_attempts - 1:  # i.e., not the last attempt
                time.sleep(0.5)  # Wait for half a second before trying again
                continue
            else:
                error_message = f"Error: cannot open image {file_path}. Reason: {e}"
                results.append((file_path, None, error_message))
                return  # Exit function if all attempts fail

    # Proceed with processing if the image was successfully opened
    try:
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            probabilities = F.softmax(outputs, dim=1)
            predicted_index = probabilities.argmax(1).item()
        class_labels = {0: 'Fresh', 1: 'Half-Fresh', 2: 'Spoiled'}
        prediction = class_labels[predicted_index]
        results.append((file_path, display_image, prediction))
    except Exception as e:
        error_message = f"Error: processing image {file_path}. Reason: {e}"
        results.append((file_path, None, error_message))

class ImageHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith(('.jpg', '.png','.jpeg')):
            classify_image(event.src_path)

def start_monitoring(path):
    event_handler = ImageHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=False)
    observer.start()

def update_results():
    dataframe_results = []
    gallery_images = []
    for file_path, image, prediction in results[-10:]:  # Show the last 10 results
        dataframe_results.append([file_path, prediction if image is not None else "Error"])
        if image is not None:
            gallery_images.append(image)  # Add image to gallery
        else:
            gallery_images.append("/home/aarabambi/pictures/test.png")  # Fallback image path

    return dataframe_results, gallery_images

iface = gr.Interface(
    fn=update_results,
    inputs=None,
    outputs=[
        gr.components.Dataframe(headers=["Image Path", "Prediction"]),
        gr.components.Gallery(label="Images")
    ],
    title="Meat Freshness Detection",
    description="Displays the classification of the most recently added images along with a history of past images.",
    live=True
)

if __name__ == "__main__":
    folder_path = '/home/aarabambi/pictures'  # Update this to your folder path
    monitoring_thread = Thread(target=start_monitoring, args=(folder_path,))
    monitoring_thread.start()
    iface.launch(allowed_paths=["/home/aarabambi/pictures"],share=True)
