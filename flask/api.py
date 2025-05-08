from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json


# 加载知识库并构建文档列表
with open("templates/train.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)
documents = [
    f"问题：{item.get('instruction', '')}\n回答：{item.get('output', '')}"
    for item in knowledge_data
]
with open("templates/ren_she.txt","r",encoding="utf-8")as f:
    re_she = f.read()
# 导入FAISS索引
index = faiss.read_index("templates/index.faiss")
# 加载模型和分词器
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model_path = f'models/Qwen3-06B'
model_name = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path)

embedding_model = SentenceTransformer("models/m3e-base")

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat_completions():
    try:
        data = request.json

        messages = data.get("messages")

        def chat_completions_model(messages, max_tokens=500, temperature=0.1):
            '''
            使用本地LLM结合RAG检索进行问答
            '''
            # 构造RAG检索
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            user_question = messages[-1]["content"]
            question_embedding = embedding_model.encode([user_question])
            D, I = index.search(question_embedding, k=3)
            related_docs = "\n---\n".join([documents[i] for i in I[0]])
            rag_prefix = (
                "请你根据以下提供的记录与用户交流。\n"
                "记录如下（可能不完全匹配，但请尽可能参考）：\n"
                f"{related_docs}\n"
                "回答时请尽量基于上述内容，并避免编造。同时在回答时要与以下人设相符\n"
                "人设如下\n"
                f"{re_she}"
            )
            system_prompt = {
                "role": "system",
                "content": rag_prefix
            }

            full_conversation = [system_prompt] + messages

            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                tokenizer.pad_token_id = tokenizer.eos_token_id

            formatted_input = tokenizer.apply_chat_template(
                full_conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(formatted_input, return_tensors='pt').to('cuda')

            input_tokens = inputs.input_ids.shape[1]

            outputs = model_name.generate(
                inputs.input_ids,
                max_length=input_tokens + max_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return response
        ai_response =  chat_completions_model(messages=messages,temperature=0.9)
        if "</think>" in ai_response:
           ai_response = ai_response.split("</think>")[-1].strip()
        else:
            ai_response = ai_response.strip()
        return jsonify({"reply": ai_response})

    except Exception as e:
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)