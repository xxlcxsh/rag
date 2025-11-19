import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYSTEM_PROMPT = (
    "Используя только предоставленный контекст, дай краткий и точный ответ на вопрос пользователя. "
    "Не придумывай информацию, если информация в контексте отсутствует, "
    "выдай типовое сообщение: 'Информация в предоставленных документах отсутствует.'. "
    "Ответ должен быть коротким, насколько это возможно, но при этом информативным"
)

def load_llm(model_path: str):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def generate_answer(tokenizer, model, question: str, context: str, system_prompt: str = SYSTEM_PROMPT) -> str:
    prompt = f"{system_prompt}\nКонтекст:\n{context}\n\nВопрос: {question}\n Ответ:"
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    gen_config = GenerationConfig(
        max_new_tokens=200,
        temperature=0.2,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
    )
    with torch.no_grad():
        output = model.generate(**inputs, generation_config=gen_config)

    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full_text.split("Ответ:", 1)[-1].strip()
    return answer
