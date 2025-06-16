# AI Personalization with Cloudflare

Chat app that automatically personalizes AI responses using Cloudflare AI and KV storage.

## What it does

- Uses Cloudflare AI for chat responses
- Stores user preferences in Cloudflare KV
- Automatically updates personalization based on feedback
- Each user gets AI responses tailored to their stored preferences

## Setup

```bash
uv sync
```

or

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
CLOUDFLARE_ACCOUNT_ID=your_account_id
KV_TOKEN=your_api_token
# token should have KV and Workers AI access
NAMESPACE_ID=your_kv_namespace_id
```

Run:

```bash
cd src
uv run app.py
#python app.py
```

## How personalization works

1. User chats → AI responds with default behavior
2. User gives feedback → System generates preference update
3. Preferences stored in Cloudflare KV by user ID
4. Future chats include stored preferences as context
5. AI responses become increasingly personalized

## License

MIT
