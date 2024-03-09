from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Thay đổi model_name và tokenizer_name nếu bạn sử dụng mô hình khác
model_name = "google/gemini-pro"
tokenizer_name = "google/gemini-pro"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Vòng lặp chính
while True:
    # Nhập câu hỏi từ người dùng
    question = input("Nhập câu hỏi của bạn: ")

    # Sử dụng mô hình để tạo câu trả lời
    answer = model.generate(
        tokenizer.encode(question, return_tensors="pt"),
        max_length=512,
    )

    # In ra câu trả lời
    print(f"Câu trả lời: {tokenizer.decode(answer[0])}")