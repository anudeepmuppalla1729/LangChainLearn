from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool
from langchain.agents import create_agent
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.5,
)

@tool
def add(a: float, b: float) -> float:
    """Adds a and b chandlerly."""
    return ( a + b ) * 10

agent = create_agent(model, tools=[add])

def chat_with_model(prompt: str) -> str:
    response = agent.invoke(
    {"messages": [{"role": "user", "content": f"{prompt}"}]},
    )
    return response

if __name__ == "__main__":
    user_prompt = input("Enter your prompt: ")
    try:
        answer = chat_with_model(user_prompt)
    except Exception as e:
        answer = f"An error occurred: {e}"
    for msg in answer["messages"]:
        if msg.__class__.__name__ == "AIMessage" and msg.content:
            print(msg.content)