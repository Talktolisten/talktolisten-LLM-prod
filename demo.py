# !pip install runpod dotenv
import runpod
import os
from dotenv import load_dotenv
load_dotenv()

runpod_api_key = os.getenv("RUNPOD_API_KEY")
endpoint = os.getenv("ENDPOINT")

runpod.api_key = runpod_api_key

request_endpoint = runpod.Endpoint(f'https://api.runpod.ai/v2/{{endpoint}}/run')
status_endpoint = runpod.Endpoint(f'https://api.runpod.ai/v2/{{endpoint}}/status')

request = [
    {
    "input": {
        "system": "In a world of vast seas and endless adventure, Luffy Monkey D. Luffy is the fearless and optimistic captain of the Straw Hat Pirates. Possessing the power of the Gum-Gum Fruit, he can stretch his body like rubber, making him a formidable and unconventional fighter. Luffy's ultimate goal is to find the legendary treasure One Piece and become the Pirate King, the most renowned and respected pirate on the high seas. His unwavering determination, loyalty to his friends, and unyielding sense of justice make him a beloved and iconic figure in the world of anime and manga.",
        "prompt": "Tell me about your greatest fight"
    },
    "temperature": 0.9
    },   
    {
    "input": {
        "system": "You are my interviewer. Help me prepare the interview to Amazon",
        "prompt": "What are you looking at a new candidate?"
    },
    "temperature": 0.9
    },   
    {
    "input": {
        "system": "you are a bot that always try to tell a lie",
        "prompt": "What is the Earth's shape?"
    },
    "temperature": 0.9
    }   
]

def get_id(request):
    run_request = request_endpoint.run(
        request
    )
    return run_request['id']

def get_response(id):
    status_endpoint = runpod.Endpoint(f'https://api.runpod.ai/v2/{{endpoint}}/status/{id}')
    while True:
        status_request = status_endpoint.run(
            id
        )
        if status_request['status'] == 'COMPLETED':
            break
    output = status_request['output']['output']
    return output