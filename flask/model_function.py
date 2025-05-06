import re
from transformers import pipeline
from transformers import TextStreamer, StoppingCriteria, StoppingCriteriaList
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json

# === 初始化嵌入模型和 FAISS 索引（建议在全局只加载一次）===
embedding_model = SentenceTransformer("models/m3e-base")

# 加载知识库并构建文档列表
with open("templates/train.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)
documents = [
    f"问题：{item.get('instruction', '')}\n回答：{item.get('output', '')}"
    for item in knowledge_data
]

# 构建 FAISS 索引
index = faiss.read_index("templates/index.faiss")

def chat_completions_model(model, messages, max_tokens=500, temperature=0.1):
    '''
    使用本地LLM结合RAG检索进行问答
    '''

    # === 1. 构造 RAG 检索 ===
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    user_question = messages[-1]["content"]
    question_embedding = embedding_model.encode([user_question])
    D, I = index.search(question_embedding, k=3)
    related_docs = "\n---\n".join([documents[i] for i in I[0]])
    rag_prefix = (
        "请你根据以下提供的知识内容回答用户的问题。\n"
        "知识来源如下（可能不完全匹配，但请尽可能参考）：\n"
        f"{related_docs}\n"
        "回答时请尽量基于上述内容，并避免编造。\n\n"
    )

    # 添加 system prompt
    system_prompt = {
        "role": "system",
        "content": rag_prefix
    }

    full_conversation = [system_prompt] + messages

    # === 2. 加载模型和分词器 ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_path = f'models/{model}'
    model_name = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # === 3. 格式化输入 ===
    formatted_input = tokenizer.apply_chat_template(
        full_conversation,
        tokenize=False,
        add_generation_prompt=True
    )
    inputs = tokenizer(formatted_input, return_tensors='pt').to('cuda')

    input_tokens = inputs.input_ids.shape[1]

    # === 4. 推理 ===
    outputs = model_name.generate(
        inputs.input_ids,
        max_length=input_tokens + max_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response.split('<｜Assistant｜>')[-1]
    total_tokens = outputs.shape[1]
    generated_tokens = total_tokens - input_tokens

    return response, input_tokens, total_tokens, generated_tokens



def related_questions_model(question,n=5):
    # === RAG 检索 ===
    question_embedding = embedding_model.encode([question])
    D, I = index.search(question_embedding, k=10)
    related_docs = "\n---\n".join([documents[i] for i in I[0]])
    rag_prefix = (
        "请你参考以下知识内容，基于原始问题生成相关问题。\n"
        "知识来源如下：\n"
        f"{related_docs}\n"
        "注意：请不要编造无关内容，只基于知识内容和原始问题进行改写。\n\n"
    )

    # === 构造 Prompt ===
    prompt = (
        f"{rag_prefix}"
        f"原始问题：{question}\n\n"
        f"请根据上面的内容生成{n}个相关提问，每个问题单独一行并用阿拉伯数字编号，不要包含其他内容。\n\n"
        "### 回答：\n"
    )

    # === 加载模型和分词器 ===
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model_path = f'models/asdasd'
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # === 配置生成管道 ===
    generator = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        return_full_text=False
    )

    # === 推理生成 ===
    output = generator(
        prompt,
        max_new_tokens=200,
        num_return_sequences=1,
        temperature=0.7
    )

    # === 清洗输出 ===
    generated_text = output[0]['generated_text']
    questions = generated_text.split('</think>')[1]
    return questions


# === RAG 检索函数 ===
def generate_reference_data(question_id):
    """
    根据用户输入检索数据库中相关的 instruction/output 数据，
    并返回 JSON 格式的结构化内容（列表）。
    """
    query_embedding = embedding_model.encode([question_id])
    D, I = index.search(query_embedding,k=10)

    related_items = [knowledge_data[i] for i in I[0]]

    result = [
        {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", "")
        }
        for item in related_items
    ]

    return result

if __name__ == '__main__':
    print(related_questions_model('解方程3x=9'))