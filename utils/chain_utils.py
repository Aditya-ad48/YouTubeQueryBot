from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
from langchain_ollama import OllamaLLM
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

def build_rag_chain(retriever,session_id):
    model = OllamaLLM(model="mistral")

    memory_store = {}

    # Function to get or create message history for a session
    def get_session_history(session_id):
        if session_id not in memory_store:
            memory_store[session_id] = ChatMessageHistory()
        return memory_store[session_id]

    prompt = PromptTemplate(
        template="""
            You are a helpful assistant.
            ONLY answer questions related to the provided transcript context.
            If the user greets you or asks something unrelated to the video, politely redirect them.

            Chat history:
            {chat_history}

            Transcript context:
            {context}

            Question: {question}
            """,
        input_variables=["chat_history", "context", "question"]
    )

    # Format chat history for the prompt
    def format_history(history):
        formatted = ""
        for msg in history.messages:
            if isinstance(msg, HumanMessage):
                formatted += f"User: {msg.content}\n"
            elif isinstance(msg, AIMessage):
                formatted += f"Assistant: {msg.content}\n"
        return formatted

    parser = StrOutputParser()

    parallel_chain = RunnableParallel({
        'context': RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
        'question': RunnableLambda(lambda x: x["question"]),
        'chat_history': RunnableLambda(lambda x: format_history(get_session_history(session_id)))
    })

    main_chain = parallel_chain | prompt | model | parser

    # Wrap the chain with message history
    chain_with_history = RunnableWithMessageHistory(
        runnable=main_chain,
        get_session_history=get_session_history,
        input_messages_key="question",
        output_messages_key="answer",
        history_messages_key="chat_history"
    )
    # Custom wrapper to save messages to history
    def save_to_history(inputs, output, session_id):
        history = get_session_history(session_id)
        history.add_user_message(inputs["question"])
        history.add_ai_message(output)
        return output

    # Return the chain with history saving
    return RunnableLambda(lambda x, config: save_to_history(x, chain_with_history.invoke(x, config=config), config["configurable"]["session_id"]))