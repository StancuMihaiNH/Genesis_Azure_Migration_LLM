



```
curl -X 'POST' \
  'http://localhost:8001/chat_stream/' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "messages": [
{"role": "user",
"content": "Tell me about IT Procurement"}
  ],
  "tags": [],
  "model": "nh-qa",
  "temperature": 0,
  "max_tokens": 0,
  "top_p": 0,
  "frequency_penalty": 0,
  "presence_penalty": 0,
  "files": []
}'
```


