import streamlit as st
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights
from torchvision.utils import draw_bounding_boxes


# Load the logo
logo = Image.open("file.png")
st.set_page_config(
    page_title="Object Detector Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the model and weights
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
categories = weights.meta["categories"]
img_preprocess = weights.transforms()

@st.cache_resource
def load_model():
    model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.1)
    model.eval()
    return model

model = load_model()

def make_prediction(img):
    img_processed = img_preprocess(img)
    prediction = model(img_processed.unsqueeze(0))
    prediction = prediction[0]
    prediction["labels"] = [categories[label] for label in prediction["labels"]]
    return prediction

def create_image_with_bboxes(img, prediction):
    img_tensor = torch.tensor(img)
    img_with_bboxes = draw_bounding_boxes(
        img_tensor, 
        boxes=prediction["boxes"], 
        labels=prediction["labels"],
        colors=["Green" if label == "person" else "red" for label in prediction["labels"]],
        width=1
    )
    img_with_bboxes_np = img_with_bboxes.detach().numpy().transpose(1, 2, 0)
    return img_with_bboxes_np


st.markdown(
    """
    <style>
    body {
        color: #333333;
        background-color: #ffffff;
    }
    .stApp {
        background-color: #ffffff; /* Set background color for the entire app */
    }
    .css-1v9w43g {
        background-color: #f0f0f0; /* Example: Set background color for sidebar */
        color: #555555;
    }
    .st-bm, .st-cn {
        background-color: #ffffff; /* Example: Set background color for main content */
    }
    .st-cb {
        color: #333333;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Dashboard layout
st.sidebar.image(logo, width=100)
st.sidebar.title("EVIDENTLY AI")

period = st.sidebar.selectbox("About", ["EVIDENTLY AI", "EVIDENTLY AI DOCUMENTATION"])
st.sidebar.markdown("---")

st.sidebar.image('obj.png', use_column_width=True)
st.sidebar.markdown(
    """
    <iframe src="https://giphy.com/embed/3o72EXEfAoFRXnzDvG" width="100%" height="200" style="border:none" frameBorder="0" class="giphy-embed" allowFullScreen></iframe>
     """,
    unsafe_allow_html=True
)
st.sidebar.markdown("---")
    

# About Documentation
if period == "EVIDENTLY AI DOCUMENTATION":
    st.title("EVIDENTLY AI")
    st.markdown("""
    EVIDENTLY AI is a project aimed at demonstrating the capabilities of object detection using Faster R-CNN with ResNet50 and FPN architecture. The dashboard allows users to upload an image and visualize object detection results with bounding boxes and object counts.
    """)

    st.header("Purpose")
    st.markdown("""
    The purpose of this project is to provide a user-friendly interface for exploring object detection capabilities using a pretrained deep learning model. It serves as a demonstration of how AI can be integrated into applications for real-world tasks like image analysis and object recognition.
    """)
    

    st.header("Model Used")
    st.markdown("""
The `fasterrcnn_resnet50_fpn_v2` model is an implementation of the Faster R-CNN (Regions with Convolutional Neural Network features) object detection framework with a ResNet-50 backbone and a Feature Pyramid Network (FPN). This architecture is designed for high-performance object detection tasks and offers improved accuracy and efficiency compared to earlier versions.
""")

    st.subheader("Architecture")
    st.markdown("""
                **Faster R-CNN** is a two-stage object detection framework:
- **Region Proposal Network (RPN)**: This network proposes regions in the image that are likely to contain objects. It uses a sliding window over the convolutional feature map to generate region proposals.
- **Region of Interest (RoI) Head**: For each proposed region, this network extracts features and performs classification and bounding box regression to refine the proposals and classify the objects within them.
""")

    st.image('diagram.jpg', caption='Architecture Diagram', use_column_width=True)


    st.header("Features")
    st.markdown("""
    - Upload images in PNG, JPG, or JPEG formats for object detection.
    - Visualize bounding boxes around detected objects with color-coded labels.
    - View counts and charts of detected object types within uploaded images.
    """)

    st.header("Technology Stack")
    st.markdown("""
    - **Backend**: PyTorch for deep learning model implementation.
    - **Frontend**: Streamlit for building interactive web applications.
    - **Visualization**: Matplotlib and Streamlit's built-in components for charts and image display.
    """)

    st.header("Contact Us")
    st.markdown("""
    For more information or questions regarding this project, please contact us at [Github](https://github.com/mynkchaudhry).
    """)

else:
    st.title("Object Detection Dashboard")
    st.markdown("""
    Welcome to the Object Detector Dashboard, a powerful tool that enables you to upload an image and detect objects within it using the Faster R-CNN ResNet50 with FPN v2 model. This dashboard provides an intuitive interface for image upload, object detection, and results visualization, including bounding boxes on images and detailed object count charts.
    """)

    st.header("Image Upload")
    st.markdown("**Upload Image Here:** You can upload your image file in PNG, JPG, or JPEG format. Once the image is uploaded, the model will automatically process it to detect objects.")

    upload = st.file_uploader(label="Upload Image Here:", type=["png", "jpg", "jpeg"])

    if upload:
        img = Image.open(upload)
        prediction = make_prediction(img)
        img_with_bbox = create_image_with_bboxes(np.array(img).transpose(2, 0, 1), prediction)

        st.header("Object Detection Results")
        st.markdown("**Bounding Box Visualization:** The uploaded image will be displayed with bounding boxes around detected objects. Bounding boxes are color-coded for easier distinction: green for 'person' and red for other objects.")

        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111)
        plt.imshow(img_with_bbox)
        plt.xticks([], [])
        plt.yticks([], [])
        ax.spines[["top", "bottom", "right", "left"]].set_visible(True)
        st.pyplot(fig, use_container_width=True)

        # Counting labels
        label_counts = {}
        for label in prediction["labels"]:
            label_counts[label] = label_counts.get(label, 0) + 1

        st.header("Label Counts")
        st.markdown("**Object Count Table:** A table listing detected object types and their counts. Provides a clear and concise overview of what objects are present in the image and their frequency.")
        st.table(pd.DataFrame(label_counts.items(), columns=['Label', 'Count']))

        st.header("Object Count Chart")
        st.markdown("**Object Count Chart:** A bar chart visualizing the counts of each detected object type. Allows for quick assessment of object distribution in the image.")
        label_df = pd.DataFrame(label_counts.items(), columns=['Label', 'Count'])
        st.bar_chart(label_df.set_index('Label'))

        st.write(prediction)
    else:
        st.info("Please upload an image to proceed.")
