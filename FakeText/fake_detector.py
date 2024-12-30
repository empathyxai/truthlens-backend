import json
import google.generativeai as genai
import os
import typing_extensions as typing


def fake_detector_model(model, text):
    class fake(typing.TypedDict):
        fake: str

    config = genai.types.GenerationConfig(
        # Only one candidate.
        candidate_count=1,
        max_output_tokens=20,
        temperature=0.30,
        response_mime_type = "application/json",
        response_schema = list[fake])
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

    fake_prompt = f'''
    You are a helpful assistantthat analyzes the given text and determines whether it is a fake or real. 
    Please provide a binary score yes or no to indicate whether the given text is fake or real. 
    Provide the binary score.
    Here is the Given text: {text}

    Give a binary score yes or no to indicate whether the given text is fake or real.
    Provide the binary score.
    '''
    try:
        response = model.generate_content(fake_prompt, generation_config = config ,safety_settings=safety_settings)
        # print(response)
        resolution = response.text
        resolution = resolution.strip()
    except Exception as e:
        print('Error: ', e)
        resolution = 'Nothing is generated, Please Try Again!'
    return resolution