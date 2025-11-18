import requests

PROXY_IP = '10.33.1.51'
PROXY_PORT = 4000
TOKEN = 'sk-1nt_HXTIyftHUWNdgbMQ3g'


# This method takes a string as an input, sends it to an AI model and returns the response as a string.
# Be aware that there is rate a limit for the number of requests per minute and the number of tokens per minute you can use.
# Example usage: print(ask_ai("Hello"))
def ask_ai(text_input):
    url = f"http://{PROXY_IP}:{PROXY_PORT}/v1/chat/completions"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {TOKEN}'
    }
    body = {
        "model": "4m",
        "messages": [{"role": "user", "content": text_input}]
    }

    try:
        response = requests.post(url, json=body, headers=headers)
        chat_completion = response.json()
        if 'error' in chat_completion:
            print(f"Error from AI service: {chat_completion['error']['message']}")
            return None
        return chat_completion['choices'][0]['message']['content']
    except requests.exceptions.RequestException as e:
        print(f"Error from AI service: {e}")
        return None
    except ValueError as e:
        print(f"Error from AI service: {e}")
        return None
