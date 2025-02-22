from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss
import numpy as np
import re


def load_book(file_path):
    with open(file_path, 'r') as f:
        text = f.read()
    return text

def split_into_chapters(book):
    chapters = re.split(r"CHAPTER", book)  # Split by chapters
    chapters = [chap.strip() for chap in chapters if chap.strip()]
    return chapters

def split_into_pages(chapter):
    pages = chapter.split('\n\n\n')
    new_pages = []
    for page in pages:
        splitted = page.split('\n')
        new_pages.append(splitted[0])
    return pages

def split_entire_book_into_pages(book):
    chapters = split_into_chapters(book)[2:]
    pages = []
    page_num = 9
    for chapter_number, chapter in enumerate(chapters):
        chapter_pages = split_into_pages(chapter)
        
        for page_number, page_content in enumerate(chapter_pages):
            # Create a dictionary for each page that includes chapter and page information
            pages.append({
                'chapter': chapter_number + 1,  # Chapter starts from 1 (adjusted from zero)
                'page': page_num, # Page starts from 9 (adjusted from zero)
                'content': page_content
            })
            page_num += 1

    return pages, chapters
    
    
def clean_data(page_data):
    
    #Split the data on consecutive newlines (by page)
    clean_content = page_data['content'].split('\n\n')
    
    #Account for starting a new chapter
    if len(clean_content[0]) < 5:
        clean_content = clean_content[2:]
    else:
        clean_content = clean_content[1:]
        
    #Remove any newlines and extra spaces and then rejoin the text
    clean_content = "".join(clean_content).replace("\n", "").strip()  
    clean_content = re.sub(r'\s+', ' ', clean_content)
    
    # Capture sentence-ending punctuation
    split_text = re.split(r'([.!?])', clean_content)
    split_text = [split_text[i] + split_text[i+1] for i in range(0, len(split_text)-1, 2)]

    return split_text

#Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

#Load the book and split it into pages
book = load_book('smaller_problems_with_philosophy.txt')
pages, chapters = split_entire_book_into_pages(book)
page_data = pages[0]

#Clean the data and put it into a list
all_sentences = []
for page in pages:
    all_sentences += clean_data(page)

# Generate embeddings for each segment (chapter or page)
def generate_embeddings(texts):
    return model.encode(texts)

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

embeddings = model.encode(all_sentences)
embeddings = np.array(embeddings, dtype=np.float32)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

qa_pipeline = pipeline("question-answering")

def answer_question(context, question):
    return qa_pipeline(question=question, context=context)

def search_pages(query, page_contents, index, top_k=3, chapter_number=None, page_range=None):
    # Filter pages by chapter or page range if specified
    if chapter_number:
        filtered_pages = [page for page in page_contents if int(page['chapter']) == chapter_number]
    elif page_range:
        start_page, end_page = page_range
        filtered_pages = [page for page in page_contents if start_page <= page['page'] <= end_page]
    else:
        filtered_pages = page_contents  # No filter, return all pages

    print(f"Filtered pages: {len(filtered_pages)}")  # Debugging filtered pages length

    # Extract content from filtered pages
    filtered_sentences = []
    for page in filtered_pages:
        sentences = clean_data(page)  # Clean and split the page into sentences
        filtered_sentences.extend(sentences)  # Add sentences from this page

    # Encode query and search for the most relevant sentences
    query_embedding = model.encode([query]).astype(np.float32)
    
    # Encode all sentences and generate embeddings
    sentence_embeddings = model.encode(filtered_sentences).astype(np.float32)
    
    # Create FAISS index for the sentences
    index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
    index.add(sentence_embeddings)
    
    # Perform the search
    distances, indices = index.search(query_embedding, top_k)
    
    # If no relevant results are returned, return an empty list
    if distances[0][0] == np.inf:  # This indicates no valid results
        print("No relevant results found!")
        return []
    
    # Get the most relevant sentences
    relevant_sentences = [filtered_sentences[idx] for idx in indices[0]]

    return relevant_sentences

query = "What is a summary of this chapter?"
relevant_pages = search_pages(query, pages, index, top_k=3, chapter_number=2)

# Check if relevant pages are empty
if not relevant_pages:
    print("No relevant pages found.")
else:
    context = " ".join(relevant_pages)
    print("Context: ", context)  # Debugging the context

    answer = answer_question(context, query)
    print("Answer: ", answer['answer'])
context = " ".join(relevant_pages)
answer = answer_question(context, query)
print("Answer: ", answer['answer'])
print("Context: ", context)
