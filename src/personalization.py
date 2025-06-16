from cloudflare_client import CloudflareClient
from dotenv import load_dotenv
import gradio as gr
import os

load_dotenv()

client = CloudflareClient(
    account_id=os.getenv("CLOUDFLARE_ACCOUNT_ID"),
    api_token=os.getenv("KV_TOKEN"),
)
namespace_id = "1cec39e4783741b1a8ea07bacb487614"

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


chat_interface = gr.ChatInterface(
    chat,
    type="messages",
    flagging_mode="manual",
    flagging_options=["Like", "Spam", "Inappropriate", "Other"],
    save_history=True
)

feedback_interface = gr.Interface(
    fn=submit_feedback,
    inputs=gr.Textbox(
        label="Your Feedback",
        placeholder="e.g., 'Please be more concise in your responses' or 'I prefer technical explanations with code examples'",
        lines=5
    ),
    outputs=gr.Textbox(
        label="Status",
        lines=8
    ),
    title="Provide Feedback",
    description="Help improve your AI assistant by providing feedback about your preferences."
)

demo = gr.TabbedInterface(
    [chat_interface, feedback_interface], 
    ["Chat", "Feedback"]
)

if __name__ == "__main__":
    demo.launch()
