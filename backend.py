from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import tensorflow as tf
import numpy as np

app = FastAPI()

@app.get("/")
def root():
    return {"Greeting": "Hello, World!"}
