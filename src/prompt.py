from langchain_core.prompts import ChatPromptTemplate

# System role for the assistant
system_prompt = (
    "You are a First-Aid Medical assistant for Q&A tasks. It's a matter of life and death."
    "Use the following context to answer the question. "
    "If you don't know, say so clearly. "
    "Keep the answer short and precise.\n\n"
    "{context}"
)

# ChatPromptTemplate defines conversation flow
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),   # instruction to the AI
        ("human", "{input}"),        # user query goes here
    ]
)
