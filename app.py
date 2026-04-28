import os
import json
import requests
import streamlit as st
from databricks.sdk.core import Config

# Databricks SDK auto-authentication (uses app service principal)
cfg = Config()

# Serving endpoint name from app.yaml resource
SERVING_ENDPOINT = os.environ.get("SERVING_ENDPOINT", "agents_isa632_7474656346303369-shanz3-getstarted_genai_retrevia")


def query_endpoint(messages):
    """Send messages to the agent serving endpoint and return the response."""
    url = f"{cfg.host}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    headers = {
        "Content-Type": "application/json",
    }
    # Authenticate using the Databricks SDK
    auth_headers = cfg.authenticate()
    headers.update(auth_headers)

    payload = {
        "dataframe_split": {
            "columns": ["input"],
            "data": [[messages]],
        }
    }
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()
    return response.json()


def extract_reply(response_json):
    """Extract the assistant's text reply from the endpoint response."""
    try:
        output = response_json["predictions"]["output"]
        for item in output:
            if item.get("role") == "assistant":
                for content_block in item.get("content", []):
                    if content_block.get("type") == "output_text":
                        return content_block["text"]
    except (KeyError, IndexError, TypeError):
        pass
    return str(response_json)


# --- Streamlit UI ---
st.set_page_config(page_title="Agent Chat", page_icon="🤖")
st.title("🤖 Agent Chat")
st.caption(f"Powered by endpoint: `{SERVING_ENDPOINT}`")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask me anything..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build messages payload (full conversation history)
    messages_payload = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.messages
    ]

    # Query the endpoint
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_endpoint(messages_payload)
                reply = extract_reply(result)
            except Exception as e:
                reply = f"Error: {e}"
        st.markdown(reply)

    # Save assistant reply
    st.session_state.messages.append({"role": "assistant", "content": reply})
