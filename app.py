from scripts.chatbot import initialize_chatbot

# Initialize the chatbot
chatbot = initialize_chatbot()

if __name__ == "__main__":
    while True:
        query = input("Enter your query: ")
        response = chatbot.run(query)
        print(f"Response: {response}")
