import json
import google.generativeai as genai
import os
import base64
import boto3
import typing_extensions as typing

s3_client = boto3.client('s3')
bucket_name = 'uploadimageak-007'  # Replace with your S3 bucket name
file_key = 'uploadedimage.jpg'

model = genai.GenerativeModel('models/MODEL_NAME')


def lambda_handler(event, context):
    
    img = event.get('image_url')
    response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
    image_data = response['Body'].read()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    result = detect_image_ai_generated(model, base64_image)
    return {
        'statusCode': 200,
        'body': {'result': result}
    }


def detect_image_ai_generated(model, image):

    config = genai.types.GenerationConfig(
    # Only one candidate.
    candidate_count=1,
    temperature=0.25,
    response_mime_type = "application/json")

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE"
    }
    ]

    prompt = "Is the Image is AI Generated? Please give the answer only in 'Yes' or 'No'."
    response = model.generate_content([{'mime_type':'image/jpeg', 'data': image}, prompt], generation_config = config, safety_settings=safety_settings)
    
    return response.text