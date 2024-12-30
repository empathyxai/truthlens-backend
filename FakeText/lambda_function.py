import json
import google.generativeai as genai
import os
import typing_extensions as typing
from fake_detector import *

model = genai.GenerativeModel('models/MODEL_NAME')


def lambda_handler(event, context):
    # print("--------- In Lambda Handler --------")
    text = event.get('text')
    result = fake_detector_model(model, text)

    return {
        'statusCode': 200,
        'body': {"result": result}
    }