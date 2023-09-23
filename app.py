
# # importing the librtaries
# from flask import Flask, request, jsonify, render_template
# import numpy as np
# import nltk
# import re
# import gensim
# from gensim.parsing.preprocessing import remove_stopwords
# from gensim import corpora
# from sklearn.feature_extraction.text import TfidfVectorizer 
# import heapq
# import fitz 
# from flask import Flask, request, jsonify

# app = Flask(__name__)
# # Function to extract text from a PDF file
# def extract_text_from_pdf(pdf_file):
#     text = ""
#     doc = fitz.open(pdf_file)
#     for page in doc:
#         text += page.get_text()
#     return text

# # Specify the PDF file path
# pdf_file_path = "Document.pdf"

# # Extract text from the PDF
# txt = extract_text_from_pdf(pdf_file_path)

# # text from wikipedia about Elon Musk
# # txt = "Elon Reeve Musk FRS (/ˈiːlɒn/ EE-lon; born June 28, 1971) is an entrepreneur and business magnate. He is the founder, CEO, and Chief Engineer at SpaceX; early stage investor,[note 1] CEO, and Product Architect of Tesla, Inc.; founder of The Boring Company; and co-founder of Neuralink and OpenAI. A centibillionaire, Musk is one of the richest people in the world.Musk was born to a Canadian mother and South African father and raised in Pretoria, South Africa. He briefly attended the University of Pretoria before moving to Canada aged 17 to attend Queen's University. He transferred to the University of Pennsylvania two years later, where he received bachelor's degrees in economics and physics. He moved to California in 1995 to attend Stanford University but decided instead to pursue a business career, co-founding the web software company Zip2 with his brother Kimbal. The startup was acquired by Compaq for $307 million in 1999. Musk co-founded online bank X.com that same year, which merged with Confinity in 2000 to form PayPal. The company was bought by eBay in 2002 for $1.5 billion.In 2002, Musk founded SpaceX, an aerospace manufacturer and space transport services company, of which he is CEO and CTO. In 2004, he joined electric vehicle manufacturer Tesla Motors, Inc. (now Tesla, Inc.) as chairman and product architect, becoming its CEO in 2008. In 2006, he helped create SolarCity, a solar energy services company that was later acquired by Tesla and became Tesla Energy. In 2015, he co-founded OpenAI, a nonprofit research company that promotes friendly artificial intelligence. In 2016, he co-founded Neuralink, a neurotechnology company focused on developing brain–computer interfaces, and founded The Boring Company, a tunnel construction company. Musk has proposed the Hyperloop, a high-speed vactrain transportation system.Musk has been the subject of criticism due to unorthodox or unscientific stances and highly publicized controversies. In 2018, he was sued for defamation by a diver who advised in the Tham Luang cave rescue; a California jury ruled in favor of Musk. In the same year, he was sued by the US Securities and Exchange Commission (SEC) for falsely tweeting that he had secured funding for a private takeover of Tesla. He settled with the SEC, temporarily stepping down from his chairmanship and accepting limitations on his Twitter usage. Musk has spread misinformation about the COVID-19 pandemic and has received criticism from experts for his other views on such matters as artificial intelligence and public transport."

# #class for preprocessing and creating word embedding
# class Preprocessing:
#     #constructor
#     def __init__(self,txt):
#         # Tokenization
#         nltk.download('punkt')  #punkt is nltk tokenizer 
#         # breaking text to sentences
#         tokens = nltk.sent_tokenize(txt) 
#         self.tokens = tokens
#         self.tfidfvectoriser=TfidfVectorizer()

#     # Data Cleaning
#     # remove extra spaces
#     # convert sentences to lower case 
#     # remove stopword
#     def clean_sentence(self, sentence, stopwords=False):
#         sentence = sentence.lower().strip()
#         sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
#         if stopwords:
#           sentence = remove_stopwords(sentence)
#         return sentence

#     # store cleaned sentences to cleaned_sentences
#     def get_cleaned_sentences(self,tokens, stopwords=False):
#         cleaned_sentences = []
#         for line in tokens:
#           cleaned = self.clean_sentence(line, stopwords)
#           cleaned_sentences.append(cleaned)
#         return cleaned_sentences

#     #do all the cleaning
#     def cleanall(self):
#         cleaned_sentences = self.get_cleaned_sentences(self.tokens, stopwords=True)
#         cleaned_sentences_with_stopwords = self.get_cleaned_sentences(self.tokens, stopwords=False)
#         # print(cleaned_sentences)
#         # print(cleaned_sentences_with_stopwords)
#         return [cleaned_sentences,cleaned_sentences_with_stopwords]

#     # TF-IDF Vectorizer
#     def TFIDF(self,cleaned_sentences):
#         self.tfidfvectoriser.fit(cleaned_sentences)
#         tfidf_vectors=self.tfidfvectoriser.transform(cleaned_sentences)
#         return tfidf_vectors

#     #tfidf for question
#     def TFIDF_Q(self,question_to_be_cleaned):
#         tfidf_vectors=self.tfidfvectoriser.transform([question_to_be_cleaned])
#         return tfidf_vectors

#     # main call function
#     def doall(self):
#         cleaned_sentences, cleaned_sentences_with_stopwords = self.cleanall()
#         tfidf = self.TFIDF(cleaned_sentences)
#         return [cleaned_sentences,cleaned_sentences_with_stopwords,tfidf]
    
#     def RetrieveAnswer(question_embedding, tfidf_vectors,method=1):
#       similarity_heap = []
#       if method==1: max_similarity = float('inf')
#       else: max_similarity = -1
#       index_similarity = -1

#      for index, embedding in enumerate(tfidf_vectors):  
#       find_similarity = AnswerMe()
#       similarity = find_similarity.answer((question_embedding).toarray(),(embedding).toarray() , method).mean()
#       if method==1:
#        heapq.heappush(similarity_heap,(similarity,index))
#       else:
#        heapq.heappush(similarity_heap,(-similarity,index))
#         return similarity_heap
    

# #class for answering the question.
# class AnswerMe:
#     #cosine similarity
#     def Cosine(self, question_vector, sentence_vector):
#         dot_product = np.dot(question_vector, sentence_vector.T)
#         denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
#         return dot_product/denominator
    
#     #Euclidean distance
#     def Euclidean(self, question_vector, sentence_vector):
#         vec1 = question_vector.copy()
#         vec2 = sentence_vector.copy()
#         if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
#         vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
#         return np.linalg.norm(vec1-vec2)

#     # main call function
#     def answer(self, question_vector, sentence_vector, method):
#         if method==1: return self.Euclidean(question_vector,sentence_vector)
#         else: return self.Cosine(question_vector,sentence_vector)


# # def RetrieveAnswer(question_embedding, tfidf_vectors,method=1):
# #   similarity_heap = []
# #   if method==1: max_similarity = float('inf')
# #   else: max_similarity = -1
# #   index_similarity = -1

# #   for index, embedding in enumerate(tfidf_vectors):  
# #     find_similarity = AnswerMe()
# #     similarity = find_similarity.answer((question_embedding).toarray(),(embedding).toarray() , method).mean()
# #     if method==1:
# #       heapq.heappush(similarity_heap,(similarity,index))
# #     else:
# #       heapq.heappush(similarity_heap,(-similarity,index))
# #   return similarity_heap


# # # Take the question as input from the user
# # user_question = input("Enter your question: ")

# # # Define the method
# # method = 2

# # # Preprocess the text
# # preprocess = Preprocessing(txt)
# # cleaned_sentences, cleaned_sentences_with_stopwords, tfidf_vectors = preprocess.doall()

# # # Clean the user's question
# # question = preprocess.clean_sentence(user_question, stopwords=True)
# # question_embedding = preprocess.TFIDF_Q(question)

# # # Retrieve and print the answer
# # similarity_heap = RetrieveAnswer(question_embedding, tfidf_vectors, method)
# # print("Question: ", user_question)

# # number_of_sentences_to_print = 2
# # while number_of_sentences_to_print > 0 and len(similarity_heap) > 0:
# #     x = similarity_heap.pop(0)
# #     print(cleaned_sentences_with_stopwords[x[1]])
# #     number_of_sentences_to_print -= 1


# class AnswerMe:
#     #cosine similarity
#     def Cosine(self, question_vector, sentence_vector):
#         dot_product = np.dot(question_vector, sentence_vector.T)
#         denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
#         return dot_product/denominator
    
#     #Euclidean distance
#     def Euclidean(self, question_vector, sentence_vector):
#         vec1 = question_vector.copy()
#         vec2 = sentence_vector.copy()
#         if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
#         vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
#         return np.linalg.norm(vec1-vec2)

#     # main call function
#     def answer(self, question_vector, sentence_vector, method):
#         if method==1: return self.Euclidean(question_vector,sentence_vector)
#         else: return self.Cosine(question_vector,sentence_vector)


# def RetrieveAnswer(question_embedding, tfidf_vectors,method=1):
#   similarity_heap = []
#   if method==1: max_similarity = float('inf')
#   else: max_similarity = -1
#   index_similarity = -1

#   for index, embedding in enumerate(tfidf_vectors):  
#     find_similarity = AnswerMe()
#     similarity = find_similarity.answer((question_embedding).toarray(),(embedding).toarray() , method).mean()
#     if method==1:
#       heapq.heappush(similarity_heap,(similarity,index))
#     else:
#       heapq.heappush(similarity_heap,(-similarity,index))
#     return similarity_heap
    

    
# #   return jsonify({'answer': answer})

# # if __name__ == '__main__':
# #     app.run(debug=True)

# # from flask import Flask, request, jsonify, render_template
# # import numpy as np
# # import nltk
# # import re
# # from gensim.parsing.preprocessing import remove_stopwords
# # from gensim import corpora
# # from sklearn.feature_extraction.text import TfidfVectorizer 
# # import heapq
# # import fitz 

# # app = Flask(__name__)

# # # Function to extract text from a PDF file
# # def extract_text_from_pdf(pdf_file):
# #     text = ""
# #     doc = fitz.open(pdf_file)
# #     for page in doc:
# #         text += page.get_text()
# #     return text

# # # Specify the PDF file path
# # pdf_file_path = "Document.pdf"

# # # Extract text from the PDF
# # txt = extract_text_from_pdf(pdf_file_path)

# # # ... (Rest of your code)

# # # Route to serve the HTML interface
# # @app.route('/')
# # def index():
# #     return render_template('index.html')

# # # API endpoint to handle question and return answers
# # @app.route('/api/answer', methods=['POST'])
# # def get_answer():
# #     question = request.json['question']

# #     # ... (Your code for processing the question and returning the answer)

# #     return jsonify({'answer': get_answer})

# # #class for answering the question.
# # class AnswerMe:
# #     #cosine similarity
# #     def Cosine(self, question_vector, sentence_vector):
# #         dot_product = np.dot(question_vector, sentence_vector.T)
# #         denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
# #         return dot_product/denominator
    
# #     #Euclidean distance
# #     def Euclidean(self, question_vector, sentence_vector):
# #         vec1 = question_vector.copy()
# #         vec2 = sentence_vector.copy()
# #         if len(vec1)<len(vec2): vec1,vec2 = vec2,vec1
# #         vec2 = np.resize(vec2,(vec1.shape[0],vec1.shape[1]))
# #         return np.linalg.norm(vec1-vec2)

# #     # main call function
# #     def answer(self, question_vector, sentence_vector, method):
# #         if method==1: return self.Euclidean(question_vector,sentence_vector)
# #         else: return self.Cosine(question_vector,sentence_vector)

# # def RetrieveAnswer(question_embedding, tfidf_vectors,method=1):
# #     similarity_heap = []
# #     if method==1: max_similarity = float('inf')
# #     else: max_similarity = -1
# #     index_similarity = -1

# #     for index, embedding in enumerate(tfidf_vectors):  
# #         find_similarity = AnswerMe()
# #         similarity = find_similarity.answer((question_embedding).toarray(),(embedding).toarray() , method).mean()
# #         if method==1:
# #             heapq.heappush(similarity_heap,(similarity,index))
# #         else:
# #             heapq.heappush(similarity_heap,(-similarity,index))
            
# #     return similarity_heap


# user_question = input("Enter your question: ")

# # Define the method
# method = 2

# # Preprocess the text
# preprocess = Preprocessing(txt)
# cleaned_sentences, cleaned_sentences_with_stopwords, tfidf_vectors = preprocess.doall()

# # Clean the user's question
# question = preprocess.clean_sentence(user_question, stopwords=True)
# question_embedding = preprocess.TFIDF_Q(question)

# # Retrieve and print the answer
# similarity_heap = RetrieveAnswer(question_embedding, tfidf_vectors, method)
# print("Question: ", user_question)

# number_of_sentences_to_print = 2
# while number_of_sentences_to_print > 0 and len(similarity_heap) > 0:
#     x = similarity_heap.pop(0)
#     print(cleaned_sentences_with_stopwords[x[1]])
#     number_of_sentences_to_print -= 1


import numpy as np
import nltk
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
import heapq
import fitz 

# Function to extract text from a PDF file
def extract_text_from_pdf(pdf_file):
    text = ""
    doc = fitz.open(pdf_file)
    for page in doc:
        text += page.get_text()
    return text

# Specify the PDF file path
pdf_file_path = "Document.pdf"

# Extract text from the PDF
txt = extract_text_from_pdf(pdf_file_path)

# class for preprocessing and creating word embedding
class Preprocessing:
    # constructor
    def __init__(self, txt):
        # Tokenization
        nltk.download('punkt')  # punkt is nltk tokenizer 
        # breaking text to sentences
        tokens = nltk.sent_tokenize(txt)
        self.tokens = tokens
        self.tfidfvectoriser = TfidfVectorizer()

    # Data Cleaning
    # remove extra spaces
    # convert sentences to lower case 
    # remove stopword
    def clean_sentence(self, sentence, stopwords=False):
        sentence = sentence.lower().strip()
        sentence = re.sub(r'[^a-z0-9\s]', '', sentence)
        if stopwords:
            sentence = remove_stopwords(sentence)
        return sentence

    # store cleaned sentences to cleaned_sentences
    def get_cleaned_sentences(self, tokens, stopwords=False):
        cleaned_sentences = []
        for line in tokens:
            cleaned = self.clean_sentence(line, stopwords)
            cleaned_sentences.append(cleaned)
        return cleaned_sentences

    # do all the cleaning
    def cleanall(self):
        cleaned_sentences = self.get_cleaned_sentences(self.tokens, stopwords=True)
        cleaned_sentences_with_stopwords = self.get_cleaned_sentences(self.tokens, stopwords=False)
        return [cleaned_sentences, cleaned_sentences_with_stopwords]

    # TF-IDF Vectorizer
    def TFIDF(self, cleaned_sentences):
        self.tfidfvectoriser.fit(cleaned_sentences)
        tfidf_vectors = self.tfidfvectoriser.transform(cleaned_sentences)
        return tfidf_vectors

    # tfidf for question
    def TFIDF_Q(self, question_to_be_cleaned):
        tfidf_vectors = self.tfidfvectoriser.transform([question_to_be_cleaned])
        return tfidf_vectors

    # main call function
    def doall(self):
        cleaned_sentences, cleaned_sentences_with_stopwords = self.cleanall()
        tfidf = self.TFIDF(cleaned_sentences)
        return [cleaned_sentences, cleaned_sentences_with_stopwords, tfidf]

# class for answering the question.
class AnswerMe:
    # cosine similarity
    def Cosine(self, question_vector, sentence_vector):
        dot_product = np.dot(question_vector, sentence_vector.T)
        denominator = (np.linalg.norm(question_vector) * np.linalg.norm(sentence_vector))
        return dot_product / denominator

    # Euclidean distance
    def Euclidean(self, question_vector, sentence_vector):
        vec1 = question_vector.copy()
        vec2 = sentence_vector.copy()
        if len(vec1) < len(vec2): vec1, vec2 = vec2, vec1
        vec2 = np.resize(vec2, (vec1.shape[0], vec1.shape[1]))
        return np.linalg.norm(vec1 - vec2)

    # main call function
    def answer(self, question_vector, sentence_vector, method):
        if method == 1: return self.Euclidean(question_vector, sentence_vector)
        else: return self.Cosine(question_vector, sentence_vector)

def RetrieveAnswer(question_embedding, tfidf_vectors, method=1):
    similarity_heap = []
    if method == 1: max_similarity = float('inf')
    else: max_similarity = -1
    index_similarity = -1

    for index, embedding in enumerate(tfidf_vectors):  
        find_similarity = AnswerMe()
        similarity = find_similarity.answer((question_embedding).toarray(), (embedding).toarray(), method).mean()
        if method == 1:
            heapq.heappush(similarity_heap, (similarity, index))
        else:
            heapq.heappush(similarity_heap, (-similarity, index))
    return similarity_heap

# Take the question as input from the user
user_question = input("Enter your question: ")

# Define the method
method = 2

# Preprocess the text
preprocess = Preprocessing(txt)
cleaned_sentences, cleaned_sentences_with_stopwords, tfidf_vectors = preprocess.doall()

# Clean the user's question
question = preprocess.clean_sentence(user_question, stopwords=True)
question_embedding = preprocess.TFIDF_Q(question)

# Retrieve and print the answer
similarity_heap = RetrieveAnswer(question_embedding, tfidf_vectors, method)
print("Question: ", user_question)

number_of_sentences_to_print = 2
while number_of_sentences_to_print > 0 and len(similarity_heap) > 0:
    x = similarity_heap.pop(0)
    print(cleaned_sentences_with_stopwords[x[1]])
    number_of_sentences_to_print -= 1
