import gradio as gr
from personalization import chat, submit_feedback

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
