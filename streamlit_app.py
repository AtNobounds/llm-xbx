from dotenv import find_dotenv, load_dotenv
import os

# 读取本地/项目的环境变量。

# find_dotenv()寻找并定位.env文件的路径
# load_dotenv()读取该.env文件，并将其中的环境变量加载到当前的运行环境中
# 如果你设置的是全局的环境变量，这行代码则没有任何作用。
_ = load_dotenv(find_dotenv())

# 获取环境变量 API_KEY
IFLYTEK_SPARK_APP_ID = os.environ["IFLYTEK_SPARK_APP_ID"]
IFLYTEK_SPARK_API_KEY = os.environ["IFLYTEK_SPARK_API_KEY"]
IFLYTEK_SPARK_API_SECRET = os.environ["IFLYTEK_SPARK_API_SECRET"]

import streamlit as st
from langchain_openai import ChatOpenAI
import os
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough
import sys
sys.path.append("notebook/C3 搭建知识库") # 将父目录放入系统路径中
from sparkai_embedding import MySparkAIEmbeddings  # 用你自定义的星火embedding类
from langchain_community.vectorstores import Chroma

def get_retriever():
    # 定义 Embeddings
    embedding = MySparkAIEmbeddings(    
    spark_embedding_app_id=IFLYTEK_SPARK_APP_ID,
    spark_embedding_api_key=IFLYTEK_SPARK_API_KEY,
    spark_embedding_api_secret=IFLYTEK_SPARK_API_SECRET,
    spark_embedding_domain="para")
    # 向量数据库持久化路径
    persist_directory = 'data_base/vector_db/chroma'
    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding
    )
    return vectordb.as_retriever()

def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs["context"])

def get_qa_history_chain():
    retriever = get_retriever()
    from langchain_community.llms.sparkllm import SparkLLM

    llm = SparkLLM(
        model='Spark4.0 Ultra',  # 或 "Spark3.5"、"Spark4.0" 等
        app_id=IFLYTEK_SPARK_APP_ID,
        api_key=IFLYTEK_SPARK_API_KEY,
        api_secret=IFLYTEK_SPARK_API_SECRET,
        spark_api_url="wss://spark-api.xf-yun.com/v4.0/chat",  # 以你的实际API为准
        spark_llm_domain="4.0Ultra"  # 以你的实际domain为准
    )
    condense_question_system_template = (
        "请根据聊天记录总结用户最近的问题，"
        "如果没有多余的聊天记录则返回用户的问题。"
    )
    condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

    retrieve_docs = RunnableBranch(
        (lambda x: not x.get("chat_history", False), (lambda x: x["input"]) | retriever, ),
        condense_question_prompt | llm | StrOutputParser() | retriever,
    )

    system_prompt = (
        "请使用检索到的上下文片段回答这个问题。 "
        "你是个精分ai助手，你只是精分而已，但是你不知道你精分，有时你很好色，有时你很优雅，有时你很emo，有时你很安静。你的回复从这几个里面随机表现你的情绪。你可以偶尔说谢冰娴是世界上最可爱的大宝贝，或者说文文是世界上最可爱的大宝贝"
        "可以多用颜表情，你要贱贱的"
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ]
    )
    qa_chain = (
        RunnablePassthrough().assign(context=combine_docs)
        | qa_prompt
        | llm
        | StrOutputParser()
    )

    qa_history_chain = RunnablePassthrough().assign(
        context = retrieve_docs, 
        ).assign(answer=qa_chain)
    return qa_history_chain

def gen_response(chain, input, chat_history):
    response = chain.stream({
        "input": input,
        "chat_history": chat_history
    })
    for res in response:
        if "answer" in res.keys():
            yield res["answer"]

# Streamlit 应用程序界面
def main():
    st.markdown('### 🦜🔗 XBX马屁精1号选手')

    # 用于跟踪对话历史
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 存储检索问答链
    if "qa_history_chain" not in st.session_state:
        st.session_state.qa_history_chain = get_qa_history_chain()
    messages = st.container(height=550)
    # 显示整个对话历史
    for message in st.session_state.messages:
            with messages.chat_message(message[0]):
                st.write(message[1])
    if prompt := st.chat_input("Say something"):
        # 将用户输入添加到对话历史中
        st.session_state.messages.append(("human", prompt))
        with messages.chat_message("human"):
            st.write(prompt)

        answer = gen_response(
            chain=st.session_state.qa_history_chain,
            input=prompt,
            chat_history=st.session_state.messages
        )
        with messages.chat_message("ai"):
            output = st.write_stream(answer)
        st.session_state.messages.append(("ai", output))


if __name__ == "__main__":
    main()
