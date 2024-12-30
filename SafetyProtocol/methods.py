import requests
import os
hf_token = os.environ['hf_token']
API_URL = os.environ['API_URL']
headers = {"Authorization": f"Bearer {hf_token}"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()


def redaction(text):
    if text:
        mod_text = []
        for i in text.split('.'):
            output = query({
                "inputs": i,
            })
            output = output[0]
            high_score_labels = [(item['label'], item['score']) for item in output if item['score'] > 0.55]

            if len(high_score_labels) > 0:
                print(high_score_labels)
                mod_text.append('[REDACTED] : ' + str(high_score_labels[0][0]) + ':' + str(high_score_labels[0][1]))
            else:
                mod_text.append(i)

        result_text = '. '.join(mod_text)
        return result_text
    else:
        return text
    
    