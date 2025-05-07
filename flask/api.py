from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_completions():
    try:
        data = request.json

        messages = data.get("messages")

        prompt_lines = []
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "user":
                prompt_lines.append(f"User: {content}")
            elif role == "assistant":
                prompt_lines.append(f"Assistant: {content}")
        prompt_lines.append("Assistant:")
        prompt_text = "\n".join(prompt_lines)

        from model_function import chat_completions_model
        ai_response,input_tokens,total_tokens,generated_tokens =  chat_completions_model(messages=prompt_text,temperature=0.9)

        result = {
            "return":ai_response
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)