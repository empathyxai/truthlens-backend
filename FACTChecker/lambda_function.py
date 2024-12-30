import json
import google.generativeai as genai
import os
import typing_extensions as typing
from functions import *
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

#### Env Variables:
GOOGLE_API_KEY = os.environ['GOOGLE_API_KEY']
QDRANT_API_KEY = os.environ['QDRANT_API_KEY']
QDRANT_HOST_URL = os.environ['QDRANT_HOST_URL']
collection_name = os.environ['collection_name']
#### CONFIGURING CLIENTS ######
genai.configure(api_key=GOOGLE_API_KEY)
embedding_key = GOOGLE_API_KEY

qdrant_client = QdrantClient(
    url=QDRANT_HOST_URL, 
    api_key=QDRANT_API_KEY,
)

######## LAMBDA HANDLER ##########
def lambda_handler(event, context):
    print("--------- In Lambda Handler --------")
    text = event.get('text')
    result = process_text(text)

    return {
        'statusCode': 200,
        'body': {'result': result}
    }

def process_text(text):
    model = genai.GenerativeModel('models/MODEL_NAME')

    document = query_vectordb(qdrant_client, text)
    if document:
        response =  grader_model(model, document, text)
        grade = json.loads(response)[0]['useful']

        if grade == 'yes':
            answer = answer_query_model(model, text, document)

            if answer:
                fact = FACT_Checker(model, text, answer, document, fact_support_prompt)

                if fact:
                    result = {'Grade Model':response, 'Answer': answer, 'FACT': fact,'DOCUMENT': document}
                    return result
                else:
                    result = {'Grade Model':response, 'Answer': answer, 'FACT': '','DOCUMENT': document}
                    return result

            else:
                result = {'Grade Model':response, 'Answer': '', 'FACT': '', 'DOCUMENT': document}

        else:
            result = {'Grade Model':response, 'Answer': '', 'FACT': '', 'DOCUMENT': document}
            return result
    else:
        response = 'Check your query, or please try again!'
        return response


# def process_flow(query_vectordb, grader_model, answer_query_model, FACT_Checker, query, vector_database, model):
#     # Searching in RAG using User Query to fetch the docs:
#     res = query_vectordb(vector_database, query)
#     if res != 'No Document Retrieved. Please try again!':
#         res = '\n'.join(res)

#         try:
#             grade = grader_model(model, res, query)
#             grade = json.loads(grade)[0]['useful']
#             if grade == 'yes':
#                 answer = answer_query_model(model, query, res)
#         except Exception as e:
#             return e, 'Grader Model Couldnt verify, Please try again!'

#         if answer != 'Nothing is generated, Please Try Again!':

#             fact = FACT_Checker(model, query, answer, res, fact_support_prompt)
#             if fact != 'Nothing is generated, Please Try Again!':
#                 return grade, answer, fact
#             else:
#                 return grade, answer
