from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    Settings
)

from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from llama_index.core.agent.workflow import FunctionAgent


# =========================================================
# 1. EMBEDDING MODEL
# =========================================================

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)


# =========================================================
# 2. BEDROCK LLM
# =========================================================

llm = BedrockConverse(
    model="anthropic.claude-3-haiku-20240307-v1:0",
    region_name="us-east-1"
)
Settings.llm = llm


# =========================================================
# 3. LOAD DOCUMENTS
# =========================================================

documents = SimpleDirectoryReader("data").load_data()

print("\nLoaded Documents:")
for doc in documents:
    print(doc.metadata.get("file_name"))


# =========================================================
# 4. CREATE VECTOR INDEX
# =========================================================

index = VectorStoreIndex.from_documents(documents)


# =========================================================
# 5. CREATE RETRIEVAL ENGINE
# =========================================================

query_engine = index.as_query_engine(
    similarity_top_k=2
)


# =========================================================
# 6. CREATE RETRIEVAL TOOL
# =========================================================

def search_documents(question: str) -> str:
    """
    Search the company documents and return
    relevant information for the user's question.
    """

    response = query_engine.query(question)

    return str(response)


# =========================================================
# 7. CREATE AGENT
# =========================================================

agent = FunctionAgent(
    tools=[search_documents],
    llm=llm,

    system_prompt="""
    You are a helpful data retrieval agent.

    You have access to company documents through
    the search_documents tool.

    When the user asks a question that requires
    information from the documents, use the
    search_documents tool.

    Do not invent information.

    If the answer is not present in the documents,
    clearly say that the information was not found.
    """
)


# =========================================================
# 8. RUN AGENT
# =========================================================

import asyncio


async def main():

    question = input("\nAsk something: ")

    response = await agent.run(question)

    print("\nAgent Answer:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())