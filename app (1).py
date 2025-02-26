import torch
import gradio as gr
import torchvision.transforms as transforms
from PIL import Image

# Load your trained model
model = torch.load("model.pth", map_location=torch.device("cpu"))
model.eval()

# Define the transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize for model input
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Function to make predictions
def predict(image):
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(image)
        prediction = torch.argmax(output, dim=1).item()  # Get class index
    return f"Predicted Class: {prediction}"

# Create Gradio Interface
iface = gr.Interface(fn=predict, inputs="image", outputs="text")

# Launch the Gradio app
iface.launch()
