import numpy as np
from RNN import RNN
from timeit import default_timer as timer    
import tkinter as tk

rnn = RNN()

def runButton():
    rnn.run(text_area)

def testButton():
    rnn.test(text_area)

root = tk.Tk()
root.title("Model")
root.geometry("800x600")

main_frame = tk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=1)

canvas = tk.Canvas(main_frame)
canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=1)

# TextArea pentru afișare log-uri/rezultate
text_area = tk.Text(root, height=10, width=80)
text_area.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
text_area.insert(tk.END, "Hello world")

button = tk.Button(root, text="Run", command=runButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

button = tk.Button(root, text="Test", command=testButton)
button.pack(side=tk.TOP, anchor='ne', padx=10, pady=10)

scrollbar_y = tk.Scrollbar(main_frame, orient=tk.VERTICAL, command=canvas.yview)
scrollbar_y.pack(side=tk.RIGHT, fill=tk.Y)

scrollbar_x = tk.Scrollbar(main_frame, orient=tk.HORIZONTAL, command=canvas.xview)
scrollbar_x.pack(side=tk.TOP, fill=tk.X)

canvas.configure(xscrollcommand=scrollbar_x.set, yscrollcommand=scrollbar_y.set)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

second_frame = tk.Frame(canvas)

canvas.create_window((0,0), window=second_frame, anchor="nw")

# def display_images():
#     if not hasattr(display_images, "image_labels"):
#         display_images.image_labels = []  # Initializează lista de widgeturi

#     kernels = get_kernels_images()
#     images = get_output_images()

#     num_cols_kernels = (len(kernels) + 1) // 2  # Numărul de coloane pentru kernels (două rânduri)
#     num_cols_images = (len(images) + 1) // 2    # Numărul de coloane pentru images (două rânduri)

#     total_images = len(kernels) + len(images)

#     # Creează widgeturi doar dacă sunt insuficiente
#     while len(display_images.image_labels) < total_images:
#         label = tk.Label(second_frame)
#         display_images.image_labels.append(label)

#     # Afișează imaginile din kernels în două rânduri
#     for i, img in enumerate(kernels):
#         row = i % 2  # Alternativ între rândul 0 și rândul 1
#         col = i // 2  # Crește coloana după fiecare două imagini
#         img_tk = ImageTk.PhotoImage(img)
#         display_images.image_labels[i].config(image=img_tk)
#         display_images.image_labels[i].image = img_tk
#         display_images.image_labels[i].grid(row=row, column=col, padx=10, pady=10)

#     # Afișează imaginile din images în două rânduri, sub kernels
#     start_index = len(kernels)  # Începe după imaginile kernels
#     for j, img in enumerate(images):
#         row = 2 + (j % 2)  # Alternativ între rândul 2 și rândul 3
#         col = j // 2  # Crește coloana după fiecare două imagini
#         img_tk = ImageTk.PhotoImage(img)
#         display_images.image_labels[start_index + j].config(image=img_tk)
#         display_images.image_labels[start_index + j].image = img_tk
#         display_images.image_labels[start_index + j].grid(row=row, column=col, padx=10, pady=10)

root.mainloop()

