from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np
import cv2
import base64
import io
from PIL import Image
import random
import os
import logging

app = FastAPI()

@app.get("/")
def root():
    return {"Greeting": "Hello, Beautiful World!"}
