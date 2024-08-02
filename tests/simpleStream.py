import requests
from enums.model import *

url = "http://localhost:8000/chat_stream/"
headers = {"Content-Type": "application/json"}
data = {
  "messages": [
    {"role": "user", "content": "What do you think about Assassin's Creed 3"},
    {"role": "assistant", "content": "Previous response if any"}
  ],
  "tags": [],
  "model": "",
  "temperature": 0.5,
  "max_tokens": 100,
  "top_p": 1.0,
  "frequency_penalty": 0.0,
  "presence_penalty": 0.0,
  "files": []
}

def gpt4_test():
    data["model"] = OpenAIModel.GPT_4.value

def gpt4o_test():
    data["model"] = OpenAIModel.GPT_4O.value

def claude3opus_test():
    data["model"] = OpenAIModel.CLAUDE_3_OPUS.value

def test_ai():
    response = requests.post(url, json=data, headers=headers, stream=True)
    full_response = ""
    for line in response.iter_lines():
        if line:
            full_response += line.decode('utf-8')

claude3opus_test()
test_ai()