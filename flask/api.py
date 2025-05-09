from flask import Flask, request, render_template,jsonify,Response
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,StoppingCriteria,StoppingCriteriaList, TextIteratorStreamer
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json
from datetime import datetime

#数据库位置
data_path = "templates/train.json"

#模型位置
model_path = 'models/Qwen3-17B'

#人设位置
re_she_path ="templates/ren_she.txt"

#RAG模型位置
rag_model_path = "models/m3e-base"

#RAG索引位置
index_path = "templates/index.faiss"

#聊天记录数据位置
ji_lu_path = "templates/ji_lu.json"

#加载数据库
with open(data_path, "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)

#整理数据
documents = [
    f"问题：{item.get('instruction', '')}\n回答：{item.get('output', '')}"
    for item in knowledge_data
]

#加载人设数据
with open(re_she_path,"r",encoding="utf-8")as f:
    re_she = f.read()

# 导入FAISS索引
index = faiss.read_index(index_path)

# 加载模型和分词器
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)


model_name = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_path)

embedding_model = SentenceTransformer(rag_model_path)

app = Flask(__name__)

@app.route('/chat', methods=['POST'])

def chat_completions():
    try:
        #获取传来的消息
        data = request.json

        messages = data.get("messages")

        def chat_completions_model(messages, max_tokens=500, temperature=0.1):

            #加载历史聊天记录
            with open(ji_lu_path, 'r', encoding='utf-8') as file:
                ji_lu = json.load(file)
            #调整数据格式
            documents_ji_lu = [
                f"问题：{item.get('instruction', '')}\n时间：{item.get('time','')}\n回答：{item.get('output', '')}"
                for item in ji_lu
            ]

            q = messages

            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]

            #在数据库中检索与问题相关的数据
            user_question = messages[-1]["content"]
            question_embedding = embedding_model.encode([user_question])
            D, I = index.search(question_embedding, k=3)
            related_docs = "\n---\n".join([documents[i] for i in I[0]])

            #在历史聊天记录中检索与问题相关的数据
            question_embedding_jl = embedding_model.encode([user_question])
            D, I = index.search(question_embedding_jl, k=3)
            related_docs_jl = "\n---\n".join([documents_ji_lu[i] for i in I[0]])

            #构建提示词
            rag_prefix = (
                "请你根据以下提供的记录与用户交流。\n"
                "记录如下，如果与用户输入不相关则不需要进行参考：\n"
                f"{related_docs}\n"
                "回答时也可以参考以下历史聊天记录，如不相关也可不参考\n"
                "聊天记录如下，其中instruction是用户的输入，time是用户输入时的时间，output是模型根据用户输入而输出的内容"
                f"{related_docs_jl}\n"
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

            #模板化数据
            formatted_input = tokenizer.apply_chat_template(
                full_conversation,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            inputs = tokenizer(formatted_input, return_tensors='pt').to('cuda')

            input_tokens = inputs.input_ids.shape[1]

            #进行推理
            outputs = model_name.generate(
                inputs.input_ids,
                max_length=input_tokens + max_tokens,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)

            return response,q

        ai_response,q =  chat_completions_model(messages=messages,temperature=0.9)

        #清理输出
        if "</think>" in ai_response:
           ai_response = ai_response.split("</think>")[-1].strip()
        else:
            ai_response = ai_response.strip()

        #保存本次聊天记录
        try:
            with open(ji_lu_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        re = jsonify(({"reply":ai_response}))

        new_entry = {
            "instruction": f"{q}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 留空或填写当前时间
            "output": f"{re}"
        }
        data.append(new_entry)

        with open(ji_lu_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        return re

    except Exception as e:
        return jsonify({"error": f"内部错误: {str(e)}"}), 500

@app.route("/")
# 渲染 HTML 页面
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    #获取网页端传来的数据
    conversation = request.json.get("conversation", [])

    if not conversation:
        return jsonify({"error": "对话内容不能为空！"}), 400

    user_question = conversation[-1]["content"]

    #在数据库中检索
    question_embedding = embedding_model.encode([user_question])
    D, I = index.search(question_embedding, k=3)
    related_docs = "\n---\n".join([documents[i] for i in I[0]])

    #在历史聊天记录中检索
    question_embedding_jl = embedding_model.encode([user_question])
    D, I = index.search(question_embedding_jl, k=3)
    related_docs_jl = "\n---\n".join([documents[i] for i in I[0]])

    #构建提示词
    rag_prefix = (
        "请你根据以下提供的记录与用户交流。\n"
        "记录如下，如果与用户输入不相关则不需要进行参考：\n"
        f"{related_docs}\n"
        "回答时也可以参考以下历史聊天记录，如不相关也可不参考\n"
        "聊天记录如下，其中instruction是用户的输入，time是用户输入时的时间，output是模型根据用户输入而输出的内容"
        f"{related_docs_jl}\n"
        "回答时请尽量基于上述内容，并避免编造。同时在回答时要与以下人设相符\n"
        "人设如下\n"
        f"{re_she}"
    )

    system_prompt = {
        "role": "system",
        "content": f"{rag_prefix}"
    }

    #构造模型输入
    full_conversation = [system_prompt] + [{"role": "user", "content": user_question}]

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 格式化输入
    formatted_input = tokenizer.apply_chat_template(
        full_conversation,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )

    try:
        device = next(model_name.parameters()).device
        inputs = tokenizer(formatted_input, return_tensors="pt").to(device)

        #生成回复
        outputs = model_name.generate(
            inputs.input_ids,
            max_length=5000,
            temperature=0.9,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        #处理输出
        if "</think>" in response:
            ai_response = response.split("</think>")[-1].strip()
        else:
            ai_response = response.strip()

        re = Response(
            json.dumps({"response": ai_response}, ensure_ascii=False),
            mimetype='application/json; charset=utf-8'
        )

        #保存当前聊天记录
        try:
            with open(ji_lu_path, 'r', encoding='utf-8') as file:
                data = json.load(file)
        except FileNotFoundError:
            data = []

        new_entry = {
            "instruction": f"{user_question}",
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),  # 留空或填写当前时间
            "output": f"{ai_response}"
        }
        data.append(new_entry)

        with open(ji_lu_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

        return re

    except Exception as e:
        return jsonify({"error": f"处理失败: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)