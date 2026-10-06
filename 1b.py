
# Design a bot using AIML.
import time

if not hasattr(time, "clock"):
    time.clock = time.perf_counter

import aiml


kernel = aiml.Kernel()


aiml_data = """
<aiml version="1.0">


    <category>
        <pattern>HELLO</pattern>
        <template>
            Hello! How can I help you?
        </template>
    </category>

    <category>
        <pattern>HI</pattern>
        <template>
            Hi! Nice to meet you.
        </template>
    </category>

    <category>
        <pattern>WHAT IS YOUR NAME</pattern>
        <template>
            My name is AIML Bot.
        </template>
    </category>

    <category>
        <pattern>WHO ARE YOU</pattern>
        <template>
            I am a chatbot designed using Artificial Intelligence Markup Language.
        </template>
    </category>

    <category>
        <pattern>HOW ARE YOU</pattern>
        <template>
            I am fine. Thank you for asking!
        </template>
    </category>


    <category>
        <pattern>WHAT IS AIML</pattern>
        <template>
            AIML stands for Artificial Intelligence Markup Language.
            It is an XML based language used to create chatbots.
        </template>
    </category>



    <category>
        <pattern>WHAT IS AI</pattern>
        <template>
            AI stands for Artificial Intelligence.
            It enables computers to perform tasks that normally require human intelligence.
        </template>
    </category>

    <category>
        <pattern>THANK YOU</pattern>
        <template>
            You're welcome!
        </template>
    </category>


    <category>
        <pattern>BYE</pattern>
        <template>
            Goodbye! Have a nice day.
        </template>
    </category>


    <category>
        <pattern>*</pattern>
        <template>
            Sorry, I don't understand that.
        </template>
    </category>

</aiml>
"""


with open("chatbot.aiml", "w") as file:
    file.write(aiml_data)

print("AIML knowledge base created successfully: chatbot.aiml")


kernel.learn("chatbot.aiml")


print()
print("======================================")
print("           AIML CHATBOT")
print("======================================")
print("Type BYE to exit.")
print()


while True:

    user_input = input("You: ").strip().upper()

    if user_input in ["BYE", "EXIT", "QUIT"]:
        print("Bot: Goodbye! Have a nice day.")
        break

    response = kernel.respond(user_input)

    print("Bot:", response)