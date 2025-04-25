import curses
from openai import OpenAI
from dotenv import load_dotenv
import os

# Initialize the curses screen


def init_screen(stdscr):
    # Clear screen
    stdscr.clear()
    stdscr.refresh()

    # Get the initial height of the window
    height, width = stdscr.getmaxyx()

    # The function to handle the streaming response
    def handle_stream(stdscr, completion):
        accumulated_content = ""
        for chunk in completion:
            content = chunk.choices[0].delta.content
            if content:
                # Accumulate the content
                accumulated_content += content

                # Clear the screen and print the updated content
                stdscr.clear()
                try:
                    stdscr.addstr(0, 0, accumulated_content)
                except curses.error:
                    # In case the content exceeds the screen size
                    pass
                stdscr.refresh()

        # Wait for user input to close the window
        stdscr.addstr("\n\nPress any key to exit.")
        stdscr.refresh()
        stdscr.getch()

    # Call the handle_stream function with the curses screen and completion
    handle_stream(stdscr, completion)


load_dotenv()  # load auth tokens from .env file

client = OpenAI(api_key=os.environ['OPENAI_KEY'])

# Define your streaming response outside the curses wrapper
completion = client.chat.completions.create(
    model="gpt-4-vision-preview",
    max_tokens=2048,
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Write two paragraphs about anything."}
    ],
    stream=True
)

# Wrap the curses application
curses.wrapper(init_screen)
