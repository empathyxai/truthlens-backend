import json
import requests
import os
from methods import *

print('import done')
hf_token = os.environ['hf_token']
API_URL = os.environ['API_URL']
headers = {"Authorization": f"Bearer {hf_token}"}


def lambda_handler(event, context):
    text = event.get('text')
    if text:
        result = redaction(text)
    else:
        result = 'No text provided'

    return {
        'statusCode': 200,
        'body': {'result': result}
    }




