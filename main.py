from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Thay đổi model_name và tokenizer_name nếu bạn sử dụng mô hình khác
model_name = "facebook/bart-base"
tokenizer_name = "facebook/bart-base"

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Danh sách các lệnh
HELLO = "hello"
ASK = "ask"
QUIT = "quit"

# Danh sách các câu hỏi và câu trả lời mẫu
QUESTIONS = {
    "xin chào": "Chào bạn! Tôi là chatbot AI.",
    "bạn là ai": "Tôi là chatbot AI, được xây dựng bằng mô hình Gemini-Ultra-Pro 1.0.",
    "tôi cần trợ giúp tìm bệnh viện, bạn giúp tôi được chứ": "Chắc chắn rồi! Bạn muốn tìm bệnh viện ở đâu?",
    "xin cho biết địa chỉ bệnh viện Bạch Mai": "Bệnh viện Bạch Mai nằm tại số 78 Giải Phóng, Đống Đa, Hai Bà Trưng, Hà Nội.",
    "bệnh viện Bạch Mai có khám vào thứ 7 không": "Có, bệnh viện Bạch Mai có khám vào thứ 7, từ 7h30 đến 11h30 và 13h30 đến 17h30.",
    "bệnh viện Bạch Mai làm việc vào khung giờ nào": "Bệnh viện Bạch Mai làm việc từ thứ Hai đến thứ Bảy, từ 7h30 đến 17h30.",
    "bệnh viện Bạch Mai có tốt không": "Bệnh viện Bạch Mai là một bệnh viện đa khoa hạng A, có uy tín và chất lượng cao.",
    "bệnh viện An Việt chữa những bệnh gì": "Bệnh viện An Việt là một bệnh viện đa khoa, chuyên sâu về các lĩnh vực như tim mạch, ung bướu, tiêu hóa, hô hấp, ...",
    "bệnh viện có khám vào thứ 7 không": "Hầu hết các bệnh viện đều có khám vào thứ 7, nhưng bạn nên liên hệ trực tiếp với bệnh viện để biết chính xác giờ làm việc.",
    "bạn có bán quần áo cho trẻ sơ sinh không": "Tôi không bán quần áo cho trẻ sơ sinh. Tôi là chatbot AI, được xây dựng để trả lời các câu hỏi và cung cấp thông tin.",
    "bạn có làm việc vào chủ nhật không": "Tôi có thể trả lời các câu hỏi của bạn 24/7.",
}

# Hàm xử lý lệnh
def handle_command(command, question=None):
    if command == HELLO:
        print(QUESTIONS["xin chào"])

    elif command == ASK:
        if question is None:
            print("Vui lòng nhập câu hỏi của bạn.")
        else:
            answer = model.generate(
                tokenizer.encode(question, return_tensors="pt"),
                max_length=512,
            )
            print(f"Câu trả lời: {tokenizer.decode(answer[0])}")

    elif command == QUIT:
        print("Tạm biệt!")
        return True

    else:
        print("Lệnh không hợp lệ. Vui lòng nhập lại.")
        return False

# Hàm trả lời câu hỏi
def answer_question(question):
    if question in QUESTIONS:
        answer = QUESTIONS[question]
    else:
        answer = model.generate(
            tokenizer.encode(question, return_tensors="pt"),
            max_length=512,
        )
        answer = tokenizer.decode(answer[0])

    print(answer)

# Vòng lặp chính
while True:
    command = input("Nhập lệnh: ").lower()
    if command in [HELLO, ASK, QUIT]:
        if command == ASK:
            question = input("Nhập câu hỏi của bạn: ")
        should_quit = handle_command(command, question)
        if should_quit:
            break
    else:
        answer_question(command)
