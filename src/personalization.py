import os
from dotenv import load_dotenv
from cloudflare_client import CloudflareClient

load_dotenv()

client = CloudflareClient(
    account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    api_token=os.getenv("KV_TOKEN"),
)
namespace_id = os.getenv("NAMESPACE_ID")

def chat(message, history, user_id="1234"):
    model = "@cf/google/gemma-3-12b-it"
    meta_prompt = client.kv.get(namespace_id, user_id)

    messages = []
    if meta_prompt:
        initial_prompt = f"The users preferences: {meta_prompt}"
        messages.append({"role": "system", "content": initial_prompt})

    if history:
        for msg in history:
            if msg.get("role") != "system":
                messages.append(msg)

    messages.append({"role": "user", "content": message})

    response = client.ai.chat_completions(
        model=model,
        messages=messages,
    )

    return response['choices'][0]['message']['content']

def submit_feedback(feedback, user_id="1234"):
    """
    Take user feedback and use it to update their meta prompt / personalization preferences
    """
    model = "@cf/google/gemma-3-12b-it"
    prompt = f"A user has provided some feedback, respond with a new 'User Preference' statement that will be added to their profile for future personalization. Feedback: {feedback}"

    response = client.ai.chat_completions(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    update = response['choices'][0]['message']['content']

    existing_preferences = client.kv.get(namespace_id, user_id)

    if existing_preferences:
        updated_preferences = existing_preferences + "\n" + update
    else:
        updated_preferences = update

    client.kv.put(namespace_id, user_id, updated_preferences)


