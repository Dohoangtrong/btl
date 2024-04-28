import tkinter as tk
from tkinter import ttk
import os

root = tk.Tk()
root.title("BTL Al")

# Function to center the window on the screen
def center_window(window, width, height):
    screen_width = window.winfo_screenwidth()
    screen_height = window.winfo_screenheight()
    x = (screen_width - width) // 2
    y = (screen_height - height) // 2
    window.geometry('{}x{}+{}+{}'.format(width, height, x, y))

# Centering the main window on the screen
center_window(root, 300, 300)

# Styling the button
style = ttk.Style()
style.configure('TButton', font=('Arial', 12))

# Function to run DFS
def run_dfs():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    dfs_path = os.path.join(current_dir, "a.py")
    os.system("python " + dfs_path)

# Function to run ID3
def run_ID3():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "b/ID3.py")
    os.system("python " + file_path)

# Function to run Collaborative_Filtering
def run_Collaborative_Filtering():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "c/Collaborative_Filtering.py")
    os.system("python " + file_path)

# Label for the title
label_title = ttk.Label(root, text="Chọn Menu", font=('Arial', 14, 'bold'))
label_title.pack(pady=10)

# Creating the button to run DFS
button_run_dfs = ttk.Button(root, text="Run DFS", command=run_dfs)
button_run_dfs.pack(pady=20)

# Creating the button to run ID3
button_run_ID3 = ttk.Button(root, text="Run ID3", command=run_ID3)
button_run_ID3.pack(pady=20)

# Creating the button to run Collaborative_Filtering
button_Collaborative_Filtering = ttk.Button(root, text="Run Collaborative_Filtering", command=run_Collaborative_Filtering)
button_Collaborative_Filtering.pack(pady=20)


root.mainloop()