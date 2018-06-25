from tkinter import *
from tkinter import ttk


root = Tk()
Label(root, text="Hello, Tkinter!").pack()
button = ttk.Button(root,text="Click Me")
button.pack()
root.mainloop()
