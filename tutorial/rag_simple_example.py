import os
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 设置OpenAI API密钥
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

def create_ragen_system():
    # 1. 加载示例知识库文件
    with open("knowledge_base.txt", "w") as f:
        f.write("""
        RAGEN是检索增强生成的缩写。
        这是一种结合了检索系统和生成式AI模型的方法。
        RAGEN首先检索相关信息，然后利用这些信息来生成更准确的回答。
        RAGEN可以减少AI模型的幻觉问题，提高回答的准确性。
        RAGEN常用于问答系统、知识库查询和内容生成等应用。
        """)
    
    loader = TextLoader("knowledge_base.txt")
    documents = loader.load()
    
    # 2. 切分文档
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # 3. 创建嵌入和向量存储
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # 4. 创建问答链
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa

def query_ragen(qa_system, query):
    """使用RAGEN系统查询信息"""
    return qa_system.run(query)

if __name__ == "__main__":
    print("初始化RAGEN系统...")
    qa_system = create_ragen_system()
    
    # 演示查询
    queries = [
        "什么是RAGEN?",
        "RAGEN有什么优势?",
        "RAGEN适用于哪些应用场景?"
    ]
    
    print("\n===== RAGEN演示 =====")
    for query in queries:
        print(f"\n问题: {query}")
        answer = query_ragen(qa_system, query)
        print(f"回答: {answer}")
