import requests
import json
from typing import Optional, Dict, Any, List, Union
from urllib.parse import quote


class CloudflareKVClient:
    """Cloudflare KV storage operations."""
    
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request and handle response."""
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get(self, namespace_id: str, key: str) -> Optional[str]:
        """Get a value by key from KV namespace."""
        encoded_key = quote(key, safe='')
        url = f"{self.base_url}/{namespace_id}/values/{encoded_key}"
        
        try:
            response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.text
        except requests.exceptions.HTTPError:
            return None
    
    def put(self, namespace_id: str, key: str, value: str, 
            expiration: Optional[int] = None, 
            expiration_ttl: Optional[int] = None,
            metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store a key-value pair in KV namespace."""
        encoded_key = quote(key, safe='')
        url = f"{self.base_url}/{namespace_id}/values/{encoded_key}"
        
        params = {}
        if expiration:
            params['expiration'] = expiration
        if expiration_ttl:
            params['expiration_ttl'] = expiration_ttl
        
        if metadata:
            files = {
                'value': (None, value),
                'metadata': (None, json.dumps(metadata))
            }
            headers = {"Authorization": f"Bearer {self.api_token}"}
            response = requests.put(url, files=files, headers=headers, params=params)
        else:
            headers = {
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "text/plain"
            }
            response = requests.put(url, data=value, headers=headers, params=params)
        
        response.raise_for_status()
        return response.json().get('success', False)
    
    def delete(self, namespace_id: str, key: str) -> bool:
        """Delete a key-value pair from KV namespace."""
        encoded_key = quote(key, safe='')
        url = f"{self.base_url}/{namespace_id}/values/{encoded_key}"
        
        response = requests.delete(url, headers={"Authorization": f"Bearer {self.api_token}"})
        response.raise_for_status()
        return response.json().get('success', False)
    
    def bulk_put(self, namespace_id: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Store multiple key-value pairs at once."""
        url = f"{self.base_url}/{namespace_id}/bulk"
        
        formatted_items = []
        for item in items:
            formatted_item = {
                'key': item['key'],
                'value': item['value']
            }
            if 'expiration' in item:
                formatted_item['expiration'] = item['expiration']
            if 'expiration_ttl' in item:
                formatted_item['expiration_ttl'] = item['expiration_ttl']
            if 'metadata' in item:
                formatted_item['metadata'] = item['metadata']
            formatted_items.append(formatted_item)
        
        response = self._make_request('PUT', url, json=formatted_items)
        return response.get('result', {})
    
    def bulk_get(self, namespace_id: str, keys: List[str], 
                 with_metadata: bool = False) -> List[Dict[str, Any]]:
        """Get multiple values by keys."""
        url = f"{self.base_url}/{namespace_id}/bulk/get"
        
        payload = {'keys': keys}
        if with_metadata:
            payload['withMetadata'] = True
        
        response = self._make_request('POST', url, json=payload)
        return response.get('result', {}).get('values', [])
    
    def bulk_delete(self, namespace_id: str, keys: List[str]) -> Dict[str, Any]:
        """Delete multiple keys at once."""
        url = f"{self.base_url}/{namespace_id}/bulk/delete"
        
        response = self._make_request('POST', url, json=keys)
        return response.get('result', {})
    
    def list_keys(self, namespace_id: str, prefix: Optional[str] = None, 
                  limit: Optional[int] = None, cursor: Optional[str] = None) -> Dict[str, Any]:
        """List keys in a namespace."""
        url = f"{self.base_url}/{namespace_id}/keys"
        
        params = {}
        if prefix:
            params['prefix'] = prefix
        if limit:
            params['limit'] = limit
        if cursor:
            params['cursor'] = cursor
        
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"}, params=params)
        response.raise_for_status()
        return response.json()
    
    def get_metadata(self, namespace_id: str, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a key."""
        encoded_key = quote(key, safe='')
        url = f"{self.base_url}/{namespace_id}/metadata/{encoded_key}"
        
        try:
            response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"})
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json().get('result')
        except requests.exceptions.HTTPError:
            return None


class CloudflareAIClient:
    """Cloudflare AI inference operations."""
    
    def __init__(self, account_id: str, api_token: str):
        self.account_id = account_id
        self.api_token = api_token
        self.base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai"
        self.openai_base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/v1"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
    
    def _make_request(self, method: str, url: str, **kwargs) -> Dict[Any, Any]:
        """Make HTTP request and handle response."""
        response = requests.request(method, url, headers=self.headers, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def run_model(self, model_name: str, inputs: Dict[str, Any]) -> Any:
        """
        Execute AI model inference.
        
        Args:
            model_name: Name of the AI model to run
            inputs: Input data for the model (varies by model type)
            
        Returns:
            Model output (format varies by model)
        """
        url = f"{self.base_url}/run/{model_name}"
        response = self._make_request('POST', url, json=inputs)
        return response.get('result')
    
    def text_classification(self, model_name: str, text: str) -> List[Dict[str, Any]]:
        """Run text classification model."""
        result = self.run_model(model_name, {"text": text})
        return result if isinstance(result, list) else []
    
    def text_generation(self, model_name: str, prompt: str, **kwargs) -> str:
        """Run text generation model."""
        inputs = {"prompt": prompt}
        inputs.update(kwargs)
        result = self.run_model(model_name, inputs)
        return result if isinstance(result, str) else str(result)
    
    def image_generation(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Run image generation model."""
        inputs = {"prompt": prompt}
        inputs.update(kwargs)
        return self.run_model(model_name, inputs)
    
    def speech_recognition(self, model_name: str, audio_data: bytes) -> str:
        """Run speech recognition model."""
        # For audio, we need to handle binary data differently
        url = f"{self.base_url}/run/{model_name}"
        headers = {"Authorization": f"Bearer {self.api_token}"}
        files = {"audio": audio_data}
        
        response = requests.post(url, headers=headers, files=files)
        response.raise_for_status()
        result = response.json().get('result')
        return result if isinstance(result, str) else str(result)
    
    def search_models(self, search: Optional[str] = None, 
                     task: Optional[str] = None, 
                     limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """Search available AI models."""
        url = f"{self.base_url}/models/search"
        params = {}
        if search:
            params['search'] = search
        if task:
            params['task'] = task
        if limit:
            params['limit'] = limit
        
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"}, params=params)
        response.raise_for_status()
        return response.json().get('result', [])
    
    def get_model_schema(self, model_name: str) -> Dict[str, Any]:
        """Get schema/documentation for a specific model."""
        url = f"{self.base_url}/models/schema"
        params = {"model": model_name}
        
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"}, params=params)
        response.raise_for_status()
        return response.json().get('result', {})
    
    def search_tasks(self, search: Optional[str] = None) -> List[Dict[str, Any]]:
        """Search available AI tasks."""
        url = f"{self.base_url}/tasks/search"
        params = {}
        if search:
            params['search'] = search
        
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"}, params=params)
        response.raise_for_status()
        return response.json().get('result', [])
    
    def list_finetunes(self) -> List[Dict[str, Any]]:
        """List available finetunes."""
        url = f"{self.base_url}/finetunes"
        response = requests.get(url, headers={"Authorization": f"Bearer {self.api_token}"})
        response.raise_for_status()
        return response.json().get('result', [])
    
    def create_finetune(self, model: str, training_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a new finetune."""
        url = f"{self.base_url}/finetunes"
        payload = {"model": model, **training_data}
        response = self._make_request('POST', url, json=payload)
        return response.get('result', {})
    
    # OpenAI Compatible Endpoints
    def chat_completions(self, messages: List[Dict[str, str]], model: str, **kwargs) -> Dict[str, Any]:
        """
        OpenAI-compatible chat completions endpoint.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model name (e.g., "@cf/meta/llama-3.1-8b-instruct")
            **kwargs: Additional parameters like temperature, max_tokens, stream, etc.
            
        Returns:
            OpenAI-format response with choices, usage, etc.
        """
        url = f"{self.openai_base_url}/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            **kwargs
        }
        response = self._make_request('POST', url, json=payload)
        return response
    
    def embeddings(self, input_text: Union[str, List[str]], model: str) -> Dict[str, Any]:
        """
        OpenAI-compatible embeddings endpoint.
        
        Args:
            input_text: Text or list of texts to embed
            model: Embedding model name (e.g., "@cf/baai/bge-large-en-v1.5")
            
        Returns:
            OpenAI-format response with embeddings data
        """
        url = f"{self.openai_base_url}/embeddings"
        payload = {
            "model": model,
            "input": input_text
        }
        response = self._make_request('POST', url, json=payload)
        return response
    
    def get_openai_base_url(self) -> str:
        """Get the OpenAI-compatible base URL for use with OpenAI SDK."""
        return self.openai_base_url


class CloudflareClient:
    """Main Cloudflare API client with KV and AI capabilities."""
    
    def __init__(self, account_id: str, api_token: str):
        """
        Initialize the Cloudflare client.
        
        Args:
            account_id: Cloudflare account ID
            api_token: Cloudflare API token with appropriate permissions
        """
        self.account_id = account_id
        self.api_token = api_token
        
        # Initialize sub-clients
        self.kv = CloudflareKVClient(account_id, api_token)
        self.ai = CloudflareAIClient(account_id, api_token)


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = CloudflareClient(
        account_id="your_account_id",
        api_token="your_api_token"
    )
    
    # KV operations
    namespace_id = "your_namespace_id"
    client.kv.put(namespace_id, "test_key", "test_value")
    value = client.kv.get(namespace_id, "test_key")
    print(f"KV Value: {value}")
    
    # AI operations
    # OpenAI-compatible chat completions
    messages = [
        {"role": "user", "content": "Make some robot noises"}
    ]
    chat_response = client.ai.chat_completions(
        messages=messages,
        model="@cf/meta/llama-3.1-8b-instruct",
        temperature=0.7,
        max_tokens=100
    )
    print(f"Chat response: {chat_response}")
    
    # OpenAI-compatible embeddings
    embeddings = client.ai.embeddings(
        input_text="I love matcha",
        model="@cf/baai/bge-large-en-v1.5"
    )
    print(f"Embeddings: {embeddings}")
    
    # Get OpenAI base URL for use with OpenAI SDK
    openai_url = client.ai.get_openai_base_url()
    print(f"OpenAI base URL: {openai_url}")
    
    # Traditional AI operations
    # Text classification
    result = client.ai.text_classification(
        "@cf/huggingface/distilbert-sst-2-int8",
        "This movie is great!"
    )
    print(f"Classification: {result}")
    
    # Text generation
    text = client.ai.text_generation(
        "@cf/meta/llama-2-7b-chat-int8",
        "Write a short story about a robot:"
    )
    print(f"Generated text: {text}")
    
    # Search models
    models = client.ai.search_models(task="text-generation", limit=5)
    print(f"Text generation models: {[m.get('name') for m in models]}")
    
    # Get model schema
    schema = client.ai.get_model_schema("@cf/meta/llama-2-7b-chat-int8")
    print(f"Model schema: {schema}")
