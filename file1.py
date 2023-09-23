# # import torch
# # from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering

# # # Load the pre-trained DistilBERT model and tokenizer
# # model_name = "distilbert-base-cased-distilled-squad"
# # tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# # model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# # # Provide a context document and a question
# # context = "Albert Einstein was a German-born theoretical physicist who developed the theory of relativity, one of the two pillars of modern physics."
# # question = "Who was Albert Einstein?"

# # # Tokenize the context and question
# # inputs = tokenizer(question, context, return_tensors="pt")

# # # Get the answer span from the model
# # start_scores, end_scores = model(**inputs)
# # start_index = torch.argmax(start_scores)
# # end_index = torch.argmax(end_scores) + 1

# # # Get the answer text from the context
# # answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index])

# # print("Question:", question)
# # print("Answer:", answer)

# import torch
# from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
# import PyPDF2
# from docx import Document

# # Load the pre-trained DistilBERT model and tokenizer
# model_name = "distilbert-base-cased-distilled-squad"
# tokenizer = DistilBertTokenizer.from_pretrained(model_name)
# model = DistilBertForQuestionAnswering.from_pretrained(model_name)

# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     with open(pdf_file, "rb") as pdf:
#         pdf_reader = PyPDF2.PdfFileReader(pdf)
#         for page in range(pdf_reader.getNumPages()):
#             text += pdf_reader.getPage(page).extractText()
#     return text

# # Function to extract text from a Word document
# def extract_text_from_docx(docx_file):
#     doc = Document(docx_file)
#     text = ""
#     for paragraph in doc.paragraphs:
#         text += paragraph.text + "\n"
#     return text

# # Provide the path to the document file (PDF or Word)
# document_file = "example.pdf"  # Replace with your document file path

# # Check the file type and extract text accordingly
# if document_file.endswith(".pdf"):
#     context = extract_text_from_pdf(document_file)
# elif document_file.endswith(".docx"):
#     context = extract_text_from_docx(document_file)
# else:
#     raise ValueError("Unsupported document format. Please provide a PDF or Word document.")

# # Provide a question
# question = "Who was Albert Einstein?"

# # Tokenize the context and question
# inputs = tokenizer(question, context, return_tensors="pt")

# # Get the answer span from the model
# start_scores, end_scores = model(**inputs)
# start_index = torch.argmax(start_scores)
# end_index = torch.argmax(end_scores) + 1

# # Get the answer text from the context
# answer = tokenizer.decode(inputs["input_ids"][0][start_index:end_index])

# print("Question:", question)
# print("Answer:", answer)