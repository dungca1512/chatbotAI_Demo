import google.generativeai as genai
import re

API_KEY = open('API_KEY.txt', 'r').read()

genai.configure(api_key=API_KEY)

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
]

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)

def is_medical_question(question):
    """
    Hàm đơn giản kiểm tra xem câu hỏi có liên quan y tế hay không.
    Sử dụng Regular Expressions
    """
    medical_keywords = ["đau", "bệnh", "chữa", "thuốc", "bác sĩ", "sức khỏe", "triệu chứng", "bệnh viện"]
    pattern = r'\b(?:' + '|'.join(medical_keywords) + r')\b'  # Tạo pattern từ medical_keywords
    return bool(re.search(pattern, question, re.IGNORECASE))

while True:
    prompt = input("Nhập câu hỏi của bạn tại đây: ")

    if is_medical_question(prompt):
        prompt_parts = [prompt]
        response = model.generate_content(prompt_parts)
        print(f"Chatbot: {response.text}")
    else:
        print("Chatbot: Câu hỏi của bạn dường như không liên quan đến lĩnh vực y tế. Vui lòng hỏi câu hỏi xoay quanh chủ đề y tế.")