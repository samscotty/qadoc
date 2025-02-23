from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable, chain
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class Context:
    question: str
    prompt: PromptTemplate
    rewriter: PromptTemplate
    retriever: BaseRetriever
    llm: Runnable


class QA:
    """Chat with your documents."""

    def __init__(self, store: VectorStore, llm: Runnable) -> None:
        self.prompt = ChatPromptTemplate.from_template(
            """Answer the question based only on the following context:
            {context}

            Question: {question}"""
        )
        rewrite_prompt = ChatPromptTemplate.from_template(
            """Provide a better search query for web search engine to answer the given question,
            end the queries with ’**’.

            Question: {question}

            Answer:"""
        )

        self._store = store
        self._retriever = store.as_retriever()
        self._llm = llm
        self._rewriter = rewrite_prompt | self._llm | _parse_rewriter_output

    def ask(self, question: str) -> str:
        """Ask a question."""
        context = Context(
            question, self.prompt, self._rewriter, self._retriever, self._llm
        )
        answer = rag.invoke(context)
        return answer.content

    def load(self, path: Path) -> None:
        """Extract documents from a PDF file."""
        loader = PyPDFLoader(path)
        documents = load_documents(loader)
        self._store.add_documents(documents)

    @classmethod
    def create(cls, embedding: str, chat: str) -> QA:
        """Create the Q&A system."""
        llm = ChatOpenAI(model=chat, temperature=0)
        store = InMemoryVectorStore(OpenAIEmbeddings(model=embedding))
        return cls(store, llm)


def load_documents(loader: BaseLoader) -> list[Document]:
    """Load and chunk text into semantically related documents."""
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_documents(documents)


@chain
def rag(context: Context) -> BaseMessage:
    """Generate LLM predictions with Retrieval-Augmented Generation (RAG).

    Retrieves the relevant documents for the question, provides them as context to the original
    prompt, and then invokes the model to generate a final output.

    """
    new_query = context.rewriter.invoke(context.question)
    docs = context.retriever.invoke(new_query)
    formatted = context.prompt.invoke({"context": docs, "question": context.question})
    return context.llm.invoke(formatted)


def _parse_rewriter_output(message: AIMessage) -> str:
    """Parse Rewrite-Retrieve-Read prompt outputs."""
    return message.content.strip('"').strip("**")
