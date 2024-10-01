import streamlit as st
from PIL import Image
import os
import shutil
from ultralytics import YOLO

def display_image(image_path, caption):
    image = Image.open(image_path)
    st.image(image, caption=caption, use_column_width=True)

model = YOLO("coco_trained_yolo.pt")
st.title("ObjectDetector")

st.write("Leveraging the advanced capabilities of the YOLOv11 model, ObjectDetector is designed to accurately identify and locate a wide range of objects within images. Trained on the extensive COCO dataset, which includes over 80 object categories, our app excels at recognizing everyday items, from people and vehicles to animals and household objects. Experience lightning-fast performance, high accuracy, and a user-friendly interface that makes object detection accessible to everyone. Simply upload your image and let ObjectDetector do the rest!")

st.write("Object Categories Detected:")
st.markdown("""
* **People:** person
* **Vehicles:** bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter
* **Animals:** bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
* **Accessories:** backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket   

* **Household items:** bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair,   
 couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair   
 drier, toothbrush
""")

st.sidebar.image("logo.jpeg", use_column_width=True)  
st.sidebar.title("ObjectDetector")
st.sidebar.write("Leveraging the advanced capabilities of the YOLOv11 model, ObjectDetector is designed to accurately identify and locate a wide range of objects within images. Trained on the extensive COCO dataset, our app excels at recognizing everyday items, from people and vehicles to animals and household objects. Experience lightning-fast performance, high accuracy, and a user-friendly interface that makes object detection accessible to everyone. Whether you're a developer, researcher, or simply curious about object detection technology, ObjectDetector is the ideal tool for your needs.")

uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    upload_dir = "uploads"
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir)

    img_path = os.path.join(upload_dir, uploaded_file.name)
    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.write("### Original Image:")
    display_image(img_path, caption="Uploaded Image")

    st.write("Detecting objects...")
    results = model(img_path, show=False, save=True)

    result_dir = "runs/detect/predict"
    saved_img_path = None
    for file_name in os.listdir(result_dir):
        if file_name.endswith(".jpg"):
            saved_img_path = os.path.join(result_dir, file_name)
            break

    if saved_img_path:
        st.success("Detection completed!")
        st.write("### Detected Objects Image:")
        display_image(saved_img_path, caption="Detected Objects")

    shutil.rmtree(result_dir)
else:
    st.write("Please upload an image for object detection.")







