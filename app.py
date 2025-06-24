import os
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import HumanMessage, AIMessage

# Load your vector search functions
from search import load_vectordb, search, clean_text

app = FastAPI()

# CORS (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend domain in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load vector DB once
vectordb = load_vectordb()

# Callback to stream tokens over WebSocket
class OutputStreamer(BaseCallbackHandler):
    def __init__(self, websocket: WebSocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs):
        await self.websocket.send_text(token)

# WebSocket route
@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()

    memory = ConversationBufferMemory(return_messages=True)
    streamer = OutputStreamer(websocket)

    llm = ChatOpenAI(
        streaming=True,
        callbacks=[streamer],
        model="gpt-4o",
        temperature=1.0,
    )

    while True:
        try:
            user_input = await websocket.receive_text()
            if not user_input.strip():
                await websocket.send_text("[ERROR: Empty input]")
                continue

            # 1. Retrieve context from your DB
            context = clean_text(search(user_input, vectordb))

            # 2. Get chat history and construct prompt with context
            chat_history = memory.load_memory_variables({})["history"]
            system_message = (
                "You are a helpful assistant. Use the context below to answer.\n\n"
                f"Context:\n{context}\n\n"
            )
            messages = [HumanMessage(content=system_message + user_input)]

            # 3. Call LLM and stream
            response = await llm.ainvoke(messages)

            # 4. Save to memory
            memory.chat_memory.add_user_message(user_input)
            memory.chat_memory.add_ai_message(response.content)

            await websocket.send_text("[DONE]")
        except Exception as e:
            await websocket.send_text(f"[ERROR: {str(e)}]")

# Run locally for testing
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8001)))
