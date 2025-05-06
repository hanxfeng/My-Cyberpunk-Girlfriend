import os
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    '''
    用于/v1/chat/completions的api
    '''
    try:
        #token检验

        req_data = request.get_json(force=True)

        model = req_data.get("model")
        messages = req_data.get("messages")
        max_tokens = req_data.get("max_tokens",500)
        temperature = req_data.get("temperature", 1)

        if not model or not messages:
            return jsonify({"error": "字段 'model' 和 'messages' 是必填项"}), 400

        model_list = os.listdir('models')
        if not model in model_list:
            return jsonify({"error":"请求的model不存在"}),400
        # 转换 messages 为 prompt
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

        # 获取ai回答
        from model_function import chat_completions_model
        ai_response,input_tokens,total_tokens,generated_tokens =  chat_completions_model(model=model,messages=prompt_text,max_tokens=max_tokens,temperature=temperature)
        #构建返回的json文件
        result = {
            "return":ai_response
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"内部错误: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)