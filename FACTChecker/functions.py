import json
import google.generativeai as genai
import os
import typing_extensions as typing
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


def query_vectordb(qdrant_client, query):

    result = qdrant_client.search(
    collection_name="COLLECTION_NAME",
    query_vector=genai.embed_content(
        model="models/EMBEDDING_MODEL",
        content=query,
        task_type="retrieval_query",
    )["embedding"],
    limit = 1)

    if result[0].payload['text']:
        return result[0].payload['text']
    else:
        'No Document Retrieved. Please try again!'




def grader_model(model, document, question):
    class Grade(typing.TypedDict):
        useful: str
    config = genai.types.GenerationConfig(
        # Only one candidate.
        candidate_count=1,
        max_output_tokens=20,
        temperature=0.30,
        response_mime_type = "application/json",
        response_schema = list[Grade])
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

    grading_prompt = f'''
    You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n

    Here is the user question: {question}

    Give a binary score yes or no to indicate whether the retrieved document is useful to resolve user question.
    Provide the binary score.
    '''
    try:
        response = model.generate_content(grading_prompt, generation_config = config ,safety_settings=safety_settings)
        # print(response)
        resolution = response.text
        resolution = resolution.strip()
    except Exception as e:
        print('Error: ', e)
        resolution = 'Nothing is generated, Please Try Again!'
    return resolution


def answer_query_model(model, query, context):

    class Answer(typing.TypedDict):
        answer: str

    config = genai.types.GenerationConfig(
        # Only one candidate.
        candidate_count=1,
        temperature=0.25,
        response_mime_type = "application/json",
        response_schema = list[Answer])

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

    answer_prompt = f'''
    Based on the following document context, provide an answer to the query.

    "Query: {query}\n"
    "Context: {context}\n\n"
    "Answer:"

    Please do not provide any external knowledge.
    In your answer, refer only to the context document. Do not employ any outside knowledge
    '''

    try:
        response = model.generate_content(answer_prompt, generation_config = config ,safety_settings=safety_settings)
        # print(response)
        resolution = response.text
        resolution = resolution.strip()
    except Exception as e:
        print('Error: ', e)
        resolution = 'Nothing is generated, Please Try Again!'
    return resolution

fact_support_prompt = '''
You are a helpful and harmless AI assistant. You will be provided with a textual context and a model-generated response.
Your task is to analyze the response sentence by sentence and classify each sentence according to its relationship with the provided context.

**Instructions:**

1. **Decompose the response into individual sentences.**
2. **For each sentence, assign one of the following labels:**
    * **`supported`**: The sentence is entailed by the given context.  Provide a supporting excerpt from the context.
    * **`unsupported`**: The sentence is not entailed by the given context. Provide an excerpt that is close but does not fully support the sentence.
    * **`contradictory`**: The sentence is falsified by the given context. Provide a contradicting excerpt from the context.
    * **`no_rad`**: The sentence does not require factual attribution (e.g., opinions, greetings, questions, disclaimers).  No excerpt is needed for this label.

3. **For each label, provide a short rationale explaining your decision.**  The rationale should be separate from the excerpt.

**Input Format:**

The input will consist of two parts, clearly separated:

* **Context:**  The textual context used to generate the response.
* **Response:** The model-generated response to be analyzed.

**Output Format:**

For each sentence in the response, output a JSON object with the following fields:

* `"sentence"`: The sentence being analyzed.
* `"label"`: One of `supported`, `unsupported`, `contradictory`, or `no_rad`.
* `"rationale"`: A brief explanation for the assigned label.
* `"excerpt"`:  A relevant excerpt from the context. Only required for `supported`, `unsupported`, and `contradictory` labels.

Output each JSON object on a new line.

**Example:**

**Input:**

```
Context: Apples are red fruits. Bananas are yellow fruits.

Response: Apples are red. Bananas are green.  Enjoy your fruit!
```

**Output:**

{"sentence": "Apples are red.", "label": "supported", "rationale": "The context explicitly states that apples are red.", "excerpt": "Apples are red fruits."}
{"sentence": "Bananas are green.", "label": "contradictory", "rationale": "The context states that bananas are yellow, not green.", "excerpt": "Bananas are yellow fruits."}
{"sentence": "Enjoy your fruit!", "label": "no_rad", "rationale": "This is a general expression and does not require factual attribution.", "excerpt": null}
'''

def FACT_Checker(model, user_request, response, context_document, fact_support_prompt):


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

    additional_prompt = f'''
    **Now, please analyze the following context and response:**

    **User Query:**
    {user_request}

    **Context:**
    {context_document}

    **Response:**
    {response}
    '''
    try:
        response = model.generate_content(fact_support_prompt + '\n' + additional_prompt, generation_config = config ,safety_settings=safety_settings)
        # print(response)
        resolution = response.text
        resolution = resolution.strip()
    except Exception as e:
        print('Error: ', e)
        resolution = 'Nothing is generated, Please Try Again!'
    return resolution

    