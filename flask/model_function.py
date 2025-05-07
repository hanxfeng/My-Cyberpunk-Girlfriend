
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import torch
import faiss
import json

#初始化嵌入模型和 FAISS 索引
#导入你自己的模型
embedding_model = SentenceTransformer("models/m3e-base")

# 加载知识库并构建文档列表
with open("templates/train.json", "r", encoding="utf-8") as f:
    knowledge_data = json.load(f)
documents = [
    f"问题：{item.get('instruction', '')}\n回答：{item.get('output', '')}"
    for item in knowledge_data
]

# 导入FAISS索引
index = faiss.read_index("templates/index.faiss")
#加载模型和分词器
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
def chat_completions_model( messages, max_tokens=500, temperature=0.1):
    '''
    使用本地LLM结合RAG检索进行问答
    '''
    #构造RAG检索
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

#测试
if __name__ == '__main__':
    print(chat_completions_model('晚安'))