import openai
import requests


# client = openai.Client(api_key="123", base_url="http://localhost:8001/v1")
# response = client.chat.completions.create(
#     model="Qwen/Qwen2.5-7B-Instruct",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "hello. how's the day?"}
#     ],
#     temperature=0.7,
#     extra_body={"lora_path": "lora0"}
# )
# print(response)

prefetch_lora_ids = {0: ["lora4", "lora5", "lora6", "lora7", "lora8"], 1: ["lora1", "lora2", "lora3"], 2: ["lora8", "lora9"]}

# prefetch_lora_ids = {0: ["lora0", "lora1", "lora2", "lora3", "lora4"]}

response = requests.post(
    "http://localhost:8001/v1/debug",
    json={"lora_ids": prefetch_lora_ids}
)
print(response.json())