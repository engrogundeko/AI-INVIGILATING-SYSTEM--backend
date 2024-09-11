import tkinter as tk
from tkinter import simpledialog
from tkinter import Label
from PIL import Image, ImageTk

def show_image_and_get_input(image_path):
    # Create the main window
    root = tk.Tk()
    root.title("Image and Input")

    # Load the image
    image = Image.open(image_path)
    image = image.resize((300, 300))  # Resize the image as needed
    photo = ImageTk.PhotoImage(image)

    # Create a label to display the image
    image_label = Label(root, image=photo)
    image_label.pack(pady=20)

    # Ask for user input
    user_input = simpledialog.askstring("Input", "Enter your input:")

    # Print user input in console or process it as needed
    print(f"User input: {user_input}")

    # Close the Tkinter window
    root.mainloop()

# Example usage
show_image_and_get_input(r"C:\Users\Admin\Desktop\AI INVIGILATING SYSTEM\ais.jpg")
        