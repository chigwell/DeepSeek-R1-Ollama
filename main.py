import streamlit as st
import requests
import json

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "deepseek-r1:8b"


def generate_text_streaming(prompt, model=MODEL):
    url = f"{OLLAMA_BASE_URL}/api/generate"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": model,
        "prompt": prompt,
        "stream": True,
    }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(data), stream=True, timeout=120)
        response.raise_for_status()

        full_response = ""
        message_placeholder = st.empty()

        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded_chunk = chunk.decode("utf-8")
                    json_chunk = json.loads(decoded_chunk)
                    if 'response' in json_chunk:
                        text_chunk = json_chunk['response']
                        full_response += text_chunk
                        message_placeholder.write(full_response)
                    elif 'done' in json_chunk and json_chunk['done']:
                        break

                except json.JSONDecodeError as e:
                    st.error(f"Error decoding JSON chunk: {e}. Chunk: {chunk}")
                    return None
                except KeyError as e:
                    st.error(f"Missing key in JSON chunk: {e}. Chunk: {decoded_chunk}")
                    return None
                except Exception as e:
                    st.error(f"An unexpected error occurred: {e}")
                    return None
        return full_response

    except requests.exceptions.RequestException as e:
        st.error(f"Error communicating with Ollama: {e}")
        return None


def main():
    st.title(f"Ollama Client {MODEL}")

    user_prompt = st.text_area("Enter your prompt:", height=200)

    if st.button("Generate"):
        if user_prompt:
            with st.spinner("Generating response..."):
                response = generate_text_streaming(user_prompt)
                if response:
                    st.write("**Done**")
        else:
            st.warning("Please enter a prompt")


if __name__ == "__main__":
    main()