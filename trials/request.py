import requests

header = {
    "Content-Type":"application/json",
    "Authorization": 'sk-I1xNML4qSn5V4JAFoZUm3rZfqfJGyTviI82PFbbFyvXP9vew'
}

post_dict = {'model': 'gpt-4o-mini', 'messages': [{'role': 'user', 'content': 'Please translate the sentence "Walk to the other end of the lobby and wait near the exit. " into English. Your translation is: '}], 'temperature': 1, 'n': 1}

r = requests.post("https://api.chatanywhere.tech/v1/chat/completions", json=post_dict, headers=header)
print(r.json()['choices'])