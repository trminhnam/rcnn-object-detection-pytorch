import os
from io import BytesIO

import cv2
import requests
import streamlit as st
import torch
import torchvision

from src.model import RCNN
from src.run import detect

st.set_page_config(
    page_title="Object Detection App with RCNN",
    # magnifying glass
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="auto",
)

st.title("Object Detection App with RCNN")
st.write("This is a simple object detection web app based on RCNN to detect VOC2012 Dataset")

# # load model
@st.cache(allow_output_mutation=True,show_spinner=True,suppress_st_warning=True)
def load_model():
    with open('classes.txt', 'r') as f:
        classes = f.read().splitlines()
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
    backbone = torchvision.models.vgg16(weights=weights)
    transforms = weights.transforms()
    
    model = RCNN(backbone.features, classes)
    model.load(os.path.join('model', 'rcnn.pth'), map_location=device)
    model = model.to(device)
    return model, transforms, classes, device

with st.spinner(f"Loading model... 💫"): 
    temp_dir = 'temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        
    model, transforms, classes, device = load_model()
    
with st.container():
    with st.form(key='my_form'):
        col1, col2 = st.columns(2)
        bytes_data = None
        
        with col1:
            st.write("Enter the url of the image to detect")
            url = st.text_input(label="URL", help="Enter the url of the image to detect")
            if url is not None and len(url) > 0:
                response = requests.get(url)
                bytes_data = BytesIO(response.content).read()
                
        with col2:
            st.write("Or upload an image")
            uploaded_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
            if uploaded_image is not None:
                bytes_data = uploaded_image.getvalue()
                
        submit_button = st.form_submit_button(label='Detect')
        
    if bytes_data is not None:
        img_path = os.path.join(temp_dir, 'test_img.jpg')
        with open(img_path, 'wb') as f:
            f.write(bytes_data)
        
        c1, c2 = st.columns(2)
        
        with c1:
            st.write("Original Image")
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            st.image(orig_img, caption="Original Image.", use_column_width='auto')
        
        with c2:
            # detection object with an icon
            with st.spinner(f"Detecting objects in the image... 🕵️‍♀️"):
                detect(model, img_path, transforms, classes, device, nms_threshold=0.3, save_dir=temp_dir)
                result_img = cv2.imread(os.path.join(temp_dir, 'result.jpg'))
                st.write("Detected Image")
                st.image(result_img, caption="Labeled Image.", use_column_width='auto')