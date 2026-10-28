"""
Image Tagger - a small Streamlit app that uses YOLO to detect objects in a photo
and show them as tags.
"""

from collections import Counter

import streamlit as st
from PIL import Image
from ultralytics import YOLO


st.set_page_config(page_title="Image Tagger", page_icon="🏷️", layout="wide")



@st.cache_resource
def load_model():

    return YOLO("yolov8m.pt")#we can use yolov8n too but since its a nano model it will not be able to detect objects as good as the medium model


model = load_model()

st.title("🏷️ Image Tagger")
st.write("Upload a photo and the app will find the objects in it and tag them.")


st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Minimum confidence",
    min_value=0.1,
    max_value=0.9,
    value=0.40,
    step=0.05,
    help="Objects the model is less sure about than this value are ignored.",
)#this for getting that slider in streamlit sidebar to set the minimum confidence for object detection 
#i have set deafult value to 0.4 but you can change it as per your requirement.


uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
   
    image = Image.open(uploaded_file).convert("RGB")#this is for opening the uploaded image and converting it to RGB format so that it can be processed by the YOLO model.

    results = model.predict(image, conf=confidence, imgsz=1280)
    result = results[0]  

    annotated_image = result.plot()[:, :, ::-1]

    left, right = st.columns(2)# this is for creating two columns in the streamlit app to show the original image and the detected objects side by side. we can also show top and bottom but i prefer side by side because it looks better and more user friendly.
    with left:
        st.subheader("Original")
        st.image(image, use_container_width=True)
    with right:
        st.subheader("Detected objects")
        st.image(annotated_image, use_container_width=True)

   
    st.subheader("Tags")

 
    detected_names = [model.names[int(c)] for c in result.boxes.cls]

    if len(detected_names) == 0:
        st.info("No objects found. Try lowering the minimum confidence in the sidebar.")#i wanted to try exception handling here but i felt just a info message would be better for the user experience so i just added this info message instead of exception handling.
    else:
        
        counts = Counter(detected_names)#this will count the number of times each object is detected in the image and store it in a dictionary where the key is the object name and the value is the count.

        tag_text = " ".join(
            f":blue-badge[{name} × {count}]" for name, count in counts.items()#shows the detected objects as tags with their count in the format of "object name × count" and also adds a blue badge around the tag for better visibility.
        )
        st.markdown(tag_text)

        st.subheader("Details")#final details of the detected objects are shown in a table format with the object name and its confidence score in percentage format.
        rows = []
        for name, conf in zip(detected_names, result.boxes.conf):
            rows.append({"Object": name, "Confidence": f"{float(conf):.0%}"})
        st.table(rows)
else:
    st.info("Upload a JPG or PNG image to get started.")
