import random
from datetime import datetime

def greet_user(name):
    print(f"Bot: Nice to meet you, {name}! How are you today? 😊")

def how_are_you_response(feeling):
    if "good" in feeling.lower() or "fine" in feeling.lower() or "well" in feeling.lower():
        return "That's wonderful to hear! Let’s have a good chat! 😄"
    elif "not" in feeling.lower() or "bad" in feeling.lower():
        return "Oh no, I’m here to cheer you up! Want to hear a joke or do some math?"
    else:
        return "Got it. I'm always here if you need to talk or have fun! 😊"

def tell_joke():
    jokes = [
        "Why don’t scientists trust atoms? Because they make up everything!",
        "Why did the computer go to the doctor? Because it had a virus!",
        "Why did the math book look sad? Because it had too many problems!"
    ]
    return random.choice(jokes)

def calculate(expression):
    try:
        result = eval(expression)
        return f"The answer is {result}"
    except:
        return "Oops, I couldn't calculate that. Try something like 5 + 3 or 10 / 2."

def show_help():
    return (
        "I can help you with:\n"
        "- Greet you and ask how you're feeling\n"
        "- Perform basic math like addition, subtraction, multiplication, and division\n"
        "- Tell you a funny joke\n"
        "- Show current date and time\n"
        "- Type 'help' to see this list again\n"
        "- Type 'exit' to quit"
    )

print("Hello! I’m SmartBot, your Python buddy.")
name = input("Bot: What's your name?\nYou: ")
greet_user(name)

feeling = input(f"{name}: ")
print("Bot:", how_are_you_response(feeling))

print("Type 'help' to see what I can do!")

while True:
    user_input = input(f"{name}: ").lower()

    if "exit" in user_input:
        print("Bot: Goodbye! Have a great day 😊")
        break
    elif "joke" in user_input:
        print("Bot:", tell_joke())
    elif "add" in user_input or "+" in user_input or "subtract" in user_input or "-" in user_input \
         or "multiply" in user_input or "*" in user_input or "divide" in user_input or "/" in user_input:
        print("Bot:", calculate(user_input.replace("add", "+").replace("subtract", "-")
                               .replace("multiply", "*").replace("divide", "/")))
    elif "time" in user_input or "date" in user_input:
        print("Bot: The current date and time is", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    elif "help" in user_input:
        print("Bot:", show_help())
    elif "how are you" in user_input:
        print("Bot: I'm just a bot, but I feel awesome when I chat with you! 😊")
    elif "thank" in user_input:
        print("Bot: You're welcome! I'm always here to help!")
    elif "ok" in user_input or "fine" in user_input or "hmm" in user_input:
        print("Bot: Got it! Let me know how I can help 😊")
    elif "laugh" in user_input or "funny" in user_input:
        print("Bot: You need a laugh? Here's a joke for you!")
        print("Bot:", tell_joke())
    elif "calculate" in user_input or "calculation" in user_input or "math" in user_input:
        print("Bot: Sure! Try saying something like 'add 5 and 3' or 'divide 10 by 2'.")
    else:
        responses = [
            "Hmm, I didn’t catch that. Try typing 'help' to see what I can do! 🤖",
            "Oops, that’s a bit unclear. Want to try rephrasing it?",
            "I’m still learning! Type 'help' for my list of abilities."
        ]
        print("Bot:", random.choice(responses))