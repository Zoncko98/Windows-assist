import tkinter as tk
from assistant import interpret_and_execute

def submit_command(input_widget, text_widget):
    """
    Process the command submitted by the user.
    
    :param input_widget: The input field in the GUI where the user enters commands.
    :param text_widget: The output field where the assistant's responses are displayed.
    """
    user_input = input_widget.get().strip()  # Get the user input
    if user_input:  # Ensure input is not empty
        response = interpret_and_execute(user_input)  # Send it to the assistant for processing
    else:
        response = "Please enter a command."

    # Display the interaction in the text area
    text_widget.insert(tk.END, f"You: {user_input}\nAssistant: {response}\n\n")
    text_widget.yview(tk.END)  # Scroll to the bottom of the text area
    input_widget.delete(0, tk.END)  # Clear the input field after processing


def clear_text(text_widget):
    """
    Clear all the text in the output area.
    
    :param text_widget: The output field to be cleared.
    """
    text_widget.delete('1.0', tk.END)


def create_gui():
    """
    Create and initialize the graphical user interface for the assistant.
    
    Uses Tkinter to create a simple interface with an input field, output text area,
    and additional controls like Clear and Submit buttons.
    """
    root = tk.Tk()  # Initialize the main GUI window
    root.title("Smart Assistant")  # Set the window title
    root.geometry("800x600")  # Set the default window size
    root.configure(bg="#33343c")  # Set the background color

    # Frame for displaying interaction output
    text_frame = tk.Frame(root, bg="#33343c")
    text_frame.pack(expand=True, fill='both', padx=20, pady=10)

    # Multi-line Text Area for displaying conversations
    text_area = tk.Text(
        text_frame,
        wrap="word",
        font=("Segoe UI", 11),
        bg="#1d1f24",
        fg="#f1f1f1",
        insertbackground="white",
        relief="flat",
        highlightthickness=0
    )
    text_area.pack(side="left", expand=True, fill="both", padx=(0, 5))

    # Scrollbar for the text area
    scrollbar = tk.Scrollbar(text_frame, command=text_area.yview)
    scrollbar.pack(side="right", fill="y")
    text_area.config(yscrollcommand=scrollbar.set)

    # Frame for user input and controls
    input_frame = tk.Frame(root, bg="#33343c")
    input_frame.pack(fill="x", padx=20, pady=5)

    # Single-line Input Field for user commands
    input_field = tk.Entry(
        input_frame,
        font=("Segoe UI", 12),
        bg="#1d1f24",
        fg="#f1f1f1",
        insertbackground="white",
        relief="flat",
        highlightthickness=1,
        highlightbackground="#5e83fa"
    )
    input_field.pack(side="left", fill="x", expand=True, padx=(0, 5), pady=5)

    # Submit Button for sending the user command
    submit_button = tk.Button(
        input_frame,
        text="Submit",
        command=lambda: submit_command(input_field, text_area),
        font=("Segoe UI", 12, "bold"),
        bg="#5e83fa",
        fg="#f1f1f1",
        relief="flat",
        activebackground="#3c5dc9",
        activeforeground="white"
    )
    submit_button.pack(side="left", padx=5)

    # Clear Button for clearing the text area
    clear_button = tk.Button(
        input_frame,
        text="Clear",
        command=lambda: clear_text(text_area),
        font=("Segoe UI", 12, "bold"),
        bg="#e04f5f",
        fg="#f1f1f1",
        relief="flat",
        activebackground="#b0444e",
        activeforeground="white"
    )
    clear_button.pack(side="right")

    # Status Label to display assistant status indicators
    status_label = tk.Label(
        root,
        text="Ready",
        font=("Segoe UI", 10, "italic"),
        bg="#33343c",
        fg="#f1f1f1"
    )
    status_label.pack(side="bottom", pady=5)

    # Keyboard shortcut: Bind Enter key to submit input
    input_field.bind("<Return>", lambda event: submit_command(input_field, text_area))

    return root  # Return the fully-initialized GUI root component