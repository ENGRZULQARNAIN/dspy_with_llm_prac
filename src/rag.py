import ujson
import dspy
from sentence_transformers import SentenceTransformer
# from core.config import ANTHROPIC_API_KEY
import os
from dotenv import load_dotenv

load_dotenv()
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
# Model configuration
lm = dspy.LM('claude-3-7-sonnet-20250219',api_key=ANTHROPIC_API_KEY)
dspy.configure(lm=lm)

#load the data
with open("./data/ragqa_arena_tech_corpus.jsonl", "r") as f:
    data = [ujson.loads(line)["text"] for line in f]



#embedding the data
embedding_model= SentenceTransformer('sentence-transformers/static-retrieval-mrl-en-v1',device="cpu")

embedder = dspy.Embedder(embedding_model.encode)


search = dspy.retrievers.Embeddings(embedder=embedder, corpus=data, k=5)



class RAG(dspy.Module):
    def __init__(self):
        self.respond = dspy.ChainOfThought('context, question -> response')

    def forward(self, question):
        context = search(question).passages
        return self.respond(context=context, question=question)
    

rag = RAG()
response = rag(question="what are high memory and low memory on linux?")

print(response.response)













