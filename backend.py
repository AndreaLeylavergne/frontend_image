from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import base64
import io
import os
import random
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model


# Paths of images and annotated masks
images_paths = {
    "image1": "dataset/images_prepped/val/0000FT_000294.png",
    "image2": "dataset/images_prepped/val/0000FT_000576.png",
    "image3": "dataset/images_prepped/val/0000FT_001016.png"
}

annotated_masks_paths = {
    "image1": "dataset/annotations_prepped_grouped/val/0000FT_000294.png",
    "image2": "dataset/annotations_prepped_grouped/val/0000FT_000576.png",
    "image3": "dataset/annotations_prepped_grouped/val/0000FT_001016.png"
}
app = FastAPI()

class ImageID(BaseModel):
    image_id: str

# Utility functions
IMAGE_ORDERING_CHANNELS_LAST = "channels_last"
IMAGE_ORDERING_CHANNELS_FIRST = "channels_first"
IMAGE_ORDERING = IMAGE_ORDERING_CHANNELS_LAST

class_colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for _ in range(5000)]

def get_colored_segmentation_image(seg_arr, n_classes, colors=class_colors):
    output_height = seg_arr.shape[0]
    output_width = seg_arr.shape[1]
    seg_img = np.zeros((output_height, output_width, 3))
    for c in range(n_classes):
        seg_arr_c = seg_arr[:, :] == c
        seg_img[:, :, 0] += ((seg_arr_c) * (colors[c][0])).astype('uint8')
        seg_img[:, :, 1] += ((seg_arr_c) * (colors[c][1])).astype('uint8')
        seg_img[:, :, 2] += ((seg_arr_c) * (colors[c][2])).astype('uint8')
    return seg_img

def get_image_array(image_input, width, height, imgNorm="sub_mean", ordering='channels_first', read_image_type=1):
    if type(image_input) is np.ndarray:
        img = image_input
    elif isinstance(image_input, str):
        if not os.path.isfile(image_input):
            raise ValueError(f"get_image_array: path {image_input} doesn't exist")
        img = cv2.imread(image_input, read_image_type)
    else:
        raise ValueError(f"get_image_array: Can't process input type {type(image_input)}")

    if imgNorm == "sub_and_divide":
        img = np.float32(cv2.resize(img, (width, height))) / 127.5 - 1
    elif imgNorm == "sub_mean":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = np.atleast_3d(img)
        means = [103.939, 116.779, 123.68]
        for i in range(min(img.shape[2], len(means))):
            img[:, :, i] -= means[i]
        img = img[:, :, ::-1]
    elif imgNorm == "divide":
        img = cv2.resize(img, (width, height))
        img = img.astype(np.float32)
        img = img / 255.0

    if ordering == 'channels_first':
        img = np.rollaxis(img, 2, 0)
    return img

def overlay_seg_image(inp_img, seg_img):
    original_h = inp_img.shape[0]
    original_w = inp_img.shape[1]
    seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)
    fused_img = (inp_img / 2 + seg_img / 2).astype('uint8')
    return fused_img

def predict(model, inp, out_fname=None, overlay_img=False, class_names=None, show_legends=False, colors=class_colors):
    assert inp is not None
    assert isinstance(inp, (np.ndarray, str)), "Input should be the CV image or the input file name"
    if isinstance(inp, str):
        inp = cv2.imread(inp)
    assert len(inp.shape) == 3, "Image should be h,w,3 "

    output_width = 304
    output_height = 208
    input_width = 608
    input_height = 416
    n_classes = 8

    x = get_image_array(inp, input_width, input_height, ordering=IMAGE_ORDERING)
    pr = model.predict(np.array([x]))[0]
    pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
    seg_img = get_colored_segmentation_image(pr, n_classes, colors=colors)

    if inp is not None:
        original_h = inp.shape[0]
        original_w = inp.shape[1]
        seg_img = cv2.resize(seg_img, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    if overlay_img:
        seg_img = overlay_seg_image(inp, seg_img)

    if out_fname is not None:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(out_fname), exist_ok=True)
        cv2.imwrite(out_fname, seg_img)

    return pr

def to_grayscale(image):
    """
    Convert an image to grayscale.
    :param image: The image to convert.
    :return: The grayscale image.
    """
    if len(np.array(image).shape) == 2:
        return image

    if len(np.array(image).shape) == 3:
        if np.array(image).shape[2] == 1:
            return np.array(image).squeeze()

        if np.array(image).shape[2] == 3:
            return np.dot(np.array(image)[...,:3], [0.2989, 0.5870, 0.1140]).squeeze()

        if np.array(image).shape[2] == 4:
            return np.dot(np.array(image)[...,:3], [0.2989, 0.5870, 0.1140]).squeeze()

    return None

def calculate_iou(array1, array2):
    """
    Calculate the intersection over union (IoU) of two arrays.
    :param array1: The first array.
    :param array2: The second array.
    :return: The IoU of the two arrays.
    """
    assert array1.shape == array2.shape, "Arrays must have the same shape"

    # Convert the arrays to binary (0 or 1)
    array1_binary = (array1 > 0).astype(int)
    array2_binary = (array2 > 0).astype(int)

    # Calculate the intersection and union of the arrays
    intersection = np.sum(array1_binary * array2_binary)
    union = np.sum(array1_binary) + np.sum(array2_binary) - intersection

    # Calculate and return the IoU
    return intersection / union if union > 0 else 0

@app.post("/predict/")
async def predict_mask(data: ImageID):
    image_id = data.image_id
    if image_id not in images_paths:
        raise HTTPException(status_code=404, detail="Image ID not found")
    
    image_path = images_paths[image_id]
    output_path = "C:/Users/andre/Projects/backend_keras/tmp/out.png"
    pr = predict(model, inp=image_path, out_fname=output_path, overlay_img=True, show_legends=True, class_names=["void", "flat", "construction","object", "nature", "sky", "human", "vehicle"])
    
    annotated_mask_path = annotated_masks_paths[image_id]
    # Read the annotated mask image in grayscale
    annotated_mask_image = cv2.imread(annotated_mask_path, cv2.IMREAD_GRAYSCALE)
    # Normalize to ensure values are between 0 and 255
    annotated_mask_image = (annotated_mask_image / annotated_mask_image.max()) * 255
    annotated_mask_image = Image.fromarray(annotated_mask_image.astype(np.uint8)).resize((256, 128))

    # Ensure the output file was created
    if not os.path.isfile(output_path):
        raise HTTPException(status_code=500, detail="Failed to create the output image")

    mask_image = Image.open(output_path)

    annotated_mask_stream = io.BytesIO()
    mask_image_stream = io.BytesIO()
    
    annotated_mask_image.save(annotated_mask_stream, format='PNG')
    mask_image.save(mask_image_stream, format='PNG')
    
    annotated_mask_stream.seek(0)
    mask_image_stream.seek(0)
    
    annotated_data_url = base64.b64encode(annotated_mask_stream.read()).decode('utf8')
    predicted_data_url = base64.b64encode(mask_image_stream.read()).decode('utf8')
    
    return JSONResponse(content={
        "annotated_mask": "data:image/png;base64," + annotated_data_url,
        "predicted_mask": "data:image/png;base64," + predicted_data_url
    })

@app.post("/evaluate/")
async def evaluate_masks(data: dict):
    annotated_mask_data = data['annotated_mask']
    predicted_mask_data = data['predicted_mask']
    
    try:
        annotated_mask = Image.open(io.BytesIO(base64.b64decode(annotated_mask_data.split(',')[1]))).resize((256, 128))
        predicted_mask = Image.open(io.BytesIO(base64.b64decode(predicted_mask_data.split(',')[1]))).resize((256, 128))

        # Convert the predicted mask to grayscale
        predicted_mask = predicted_mask.convert("L")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error decoding images: {str(e)}")
    
    annotated_mask_array = np.array(annotated_mask)
    predicted_mask_array = np.array(predicted_mask)
    iou_score = calculate_iou(annotated_mask_array, predicted_mask_array)
    
    annotated_mask_stream = io.BytesIO()
    mask_image_stream = io.BytesIO()
    
    annotated_mask.save(annotated_mask_stream, format='PNG')
    predicted_mask.save(mask_image_stream, format='PNG')
    
    annotated_mask_stream.seek(0)
    mask_image_stream.seek(0)
    
    annotated_data_url = base64.b64encode(annotated_mask_stream.read()).decode('utf8')
    predicted_data_url = base64.b64encode(mask_image_stream.read()).decode('utf8')
    
    return JSONResponse(content={
        "iou_score": iou_score,
        "annotated_mask": "data:image/png;base64," + annotated_data_url,
        "predicted_mask": "data:image/png;base64," + predicted_data_url
    })


@app.get("/")
def root():
    return {"Greeting": "Hello, loving!"}
