
# RAG-based Tutor: NLP Concepts from Jurafsky & Martin’s Speech & Language Processing

## Project Overview

This project implements a **Retrieval-Augmented Generation (RAG)-based Tutor** to help students learn concepts from *Speech & Language Processing* by Jurafsky & Martin. The tutor retrieves relevant excerpts from the indexed textbook and generates detailed answers to student questions using a large language model (LLM).

## Features

- **Text Retrieval**: Uses FAISS to index text chunks from the book and retrieve relevant excerpts based on student queries.
- **Augmented Answer Generation**: Uses a pre-trained LLM (GPT-2/Falcon) to generate answers by augmenting the retrieved text with the user's question.
- **Provenance Display**: Shows the retrieved text excerpts used to generate the answer, ensuring transparency and reference to the original text.

## Project Structure

```
├── index.py                # Code for indexing the textbook using FAISS and generating embeddings
├── query.py                # Code for retrieving text chunks and generating responses using the LLM
├── model.py                # LLM model loading and interaction (GPT-2/Falcon)
├── requirements.txt        # Dependencies
├── README.md               # Project documentation (this file)
```

## Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/rag-based-tutor.git
   cd rag-based-tutor
   ```

2. **Set up a virtual environment** (optional but recommended):

   ```bash
   python3 -m venv env
   source env/bin/activate
   ```

3. **Install the required packages**:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Index the Text**:
   To index the textbook (replace with *Speech & Language Processing* by Jurafsky & Martin), run:

   ```bash
   python index.py
   ```

   This will preprocess the text, split it into chunks, generate embeddings using Sentence-BERT, and store them using FAISS.

2. **Ask a Question**:
   To retrieve relevant text chunks and generate an answer to a student’s question, run:

   ```bash
   python query.py --question "What is tokenization in NLP?"
   ```

   This will output a generated answer along with the relevant text excerpts retrieved from the indexed book.

## Example

```bash
Question: What is tokenization in NLP?
Relevant Information: 
"Tokenization is the process of breaking down text into individual words or phrases, known as tokens. These tokens form the building blocks of NLP tasks."
Answer: 
"Tokenization is a fundamental process in natural language processing (NLP) that involves breaking text into smaller units such as words or phrases."
```

## Models Used

- **Sentence-BERT** for text chunk embeddings.
- **GPT-2** or **Falcon-7b-instruct** for augmented answer generation.

## Contributing

Feel free to contribute by forking the repository and submitting a pull request. Any suggestions for improvements or bug fixes are welcome!

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---
