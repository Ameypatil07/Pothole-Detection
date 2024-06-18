import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import functional as F
from model import PotholeModel  # Assuming you have defined the model in a separate file
import matplotlib.pyplot as plt

IMG_SIZE = 640  # Define the size of the images

# Function to calculate perceived focal length
def calculate_perceived_focal_length(bbox):
    length = bbox[3] - bbox[1]
    width = bbox[2] - bbox[0]
    pixel_length = length  # Assuming length represents pixel length
    camera_distance = 90  # Fixed camera distance in centimeters
    return (pixel_length * camera_distance) / width

# Function to estimate dimensions
def estimate_dimensions(image, model):
    with torch.no_grad():
        # Preprocess the image
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        image_tensor = F.to_tensor(image).unsqueeze(0)

        # Perform inference
        pred_bboxes = model(image_tensor)
        
        # Calculate perceived focal length for each detected pothole
        perceived_focal_lengths = []
        for pred_bbox in pred_bboxes:
            perceived_focal_length = calculate_perceived_focal_length(pred_bbox)
            perceived_focal_lengths.append(perceived_focal_length)

        # Calculate average perceived focal length
        average_perceived_focal_length = torch.mean(torch.tensor(perceived_focal_lengths)).item()

        return average_perceived_focal_length, pred_bboxes

# Load the best model
model = PotholeModel()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()

# Function to calculate dimension estimation
def calculate_dimensions(bbox, perceived_focal_length):
    # Extract bounding box coordinates
    xmin, ymin, xmax, ymax = bbox.squeeze().tolist()

    # Calculate length and width of the bounding box
    length = ymax - ymin
    width = xmax - xmin

    # Calculate area of the pothole
    area = length * width

    # Estimate actual dimensions using the perceived focal length
    actual_length = (perceived_focal_length * length) / IMG_SIZE  # Assuming image size is used for scaling
    actual_width = (perceived_focal_length * width) / IMG_SIZE

    return actual_length, actual_width, area

# Function to compare plots with dimension estimation
def compare_plots_with_dimension(image, out_bbox, perceived_focal_length):
    # Perform dimension estimation
    actual_length, actual_width, area = calculate_dimensions(out_bbox, perceived_focal_length)

    # Plot the image with bounding boxes and dimension estimation
    fig, ax = plt.subplots()
    ax.imshow(image)

    # Plot detected bounding box
    out_xmin, out_ymin, out_xmax, out_ymax = out_bbox.squeeze().tolist()
    out_length = out_ymax - out_ymin
    out_width = out_xmax - out_xmin
    ax.add_patch(plt.Rectangle((out_xmin, out_ymin), out_width, out_length, edgecolor='r', facecolor='none', linewidth=2))
    ax.text(out_xmin, out_ymax + 20, f'Estimated Area: {area:.2f}', color='r', fontsize=10)
    ax.text(out_xmin, out_ymax + 40, f'Estimated Length: {actual_length:.2f} cm', color='r', fontsize=10)
    ax.text(out_xmin, out_ymax + 60, f'Estimated Width: {actual_width:.2f} cm', color='r', fontsize=10)

    st.pyplot(fig)

# Streamlit app
st.title("Pothole Dimension Estimation")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    image_array = np.array(image)

    if st.button('Estimate Dimensions'):
        # Estimate dimensions
        average_perceived_focal_length, pred_bboxes = estimate_dimensions(image, model)

        for pred_bbox in pred_bboxes:
            compare_plots_with_dimension(image_array, pred_bbox, average_perceived_focal_length)

        st.write(f"Estimated Average Perceived Focal Length: {average_perceived_focal_length} cm")
