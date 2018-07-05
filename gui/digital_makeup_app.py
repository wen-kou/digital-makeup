import os
import cv2
import numpy as np
import time
import copy

from PIL import Image as PILIm
from PIL import ImageTk as PILImTk
from tkinter import *
from tkinter import PhotoImage
from tkinter import filedialog
from tkinter import messagebox
from pipelines import makeup_by_whole_face_transfer


def proc_image(image, max_size):
    scale = max_size / np.max([image.shape[0], image.shape[1]])
    image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PILIm.fromarray(image)
    image = PILImTk.PhotoImage(image)
    # image = PhotoImage(image)
    return image


class DigitalMakeupApp(Frame):
    def __init__(self, master, example_image_folder_path):
        Frame.__init__(self, master)
        self.grid()

        # initialize all variables first
        self.example_image_folder_path = example_image_folder_path
        self.max_size = 500
        self.target_image_path = None
        self.origin_image_label = None
        self.makeup_image_label = None
        self.makeup_result = None
        self.selection = StringVar()
        self.launch = True

        self.create_widgets()

    def create_widgets(self):
        # Label(self, text='Digital Makeup UI').grid(row=0, column=0, sticky=W)
        example_templates_path = os.listdir(self.example_image_folder_path)
        Button(self, text="browse", command=self.load_file, width=10) \
            .grid(row=1, column=0, sticky=W)
        Button(self, text="save", command=self._save, width=10) \
            .grid(row=1, column=1, sticky=W)
        for i, example_template_path in enumerate(example_templates_path):
            current_example_selection = os.path.join(self.example_image_folder_path, example_template_path)
            if os.path.isdir(current_example_selection) is False:
                continue
            image_name = current_example_selection.split('/')[-1]
            image = cv2.imread(os.path.join(current_example_selection, 'face_' + str(image_name) + '.jpg'))
            image = proc_image(image, 40)
            Radiobutton(self,
                        variable=self.selection,
                        value=current_example_selection,
                        image=image,
                        text='Example' + str(i),
                        command=self._makeup) \
                .grid(row=3, column=i * 2, sticky=W)
            icon = Label(image=image)
            icon.image = image
        self.launch = False

    def _save(self):
        if self.makeup_result is None:
            return
        save_path = '../results'
        if os.path.exists(save_path) is False:
            os.mkdir(save_path)
        timestamp = time.time()
        timestamp = str(timestamp).split('.')[0] + '.jpg'
        save_name = os.path.join(save_path, timestamp)
        cv2.imwrite(save_name, self.makeup_result)

    def _makeup(self):

        if self.target_image_path is None:
            if self.launch is False:
                messagebox.showwarning('No valid image', 'please select an image')
            return
        print('selection is {}'.format(self.selection.get()))
        selection_name = self.selection.get().split('/')[-1]

        example_face_path = os.path.join(self.selection.get(),
                                         'face_' + selection_name + '.jpg')
        example_alpha_path = os.path.join(self.selection.get(),
                                          'face_alpha_' + selection_name + '.jpg')
        example_landmarks_path = os.path.join(self.selection.get(), selection_name + '.txt')
        example_face = cv2.imread(example_face_path)
        example_alpha_face = cv2.imread(example_alpha_path)
        example_landmarks = np.loadtxt(example_landmarks_path, dtype=int)
        target_image = cv2.imread(self.target_image_path)

        start = time.time()
        print('start to makeup')
        makeup_result = makeup_by_whole_face_transfer.makeup_by_whole_face_transfer(target_image,
                                                                                    example_face,
                                                                                    example_alpha_face,
                                                                                    example_landmarks
                                                                                    )
        print('finish makeup with process time {}s'.format(time.time() - start))
        self.makeup_result = makeup_result
        self.display_image(makeup_result, 'makeup')

    def load_file(self):
        file_name = filedialog.askopenfilename(initialdir=os.getcwd(), title="Select file",
                                               filetypes=(("jpeg files", "*.jpg"), ("jpeg files", "*.jpeg")))
        self.target_image_path = file_name
        if file_name:
            try:
                image = cv2.imread(filename=file_name)
                self.display_image(image, 'origin')
            except Exception as e:
                s = str(e)
                print(s)
                messagebox.showerror("Open Source File", "Failed to open image filed \n'%s'" % s)
                return

    def display_image(self, image, display_selection):
        photo = copy.copy(image)
        photo = proc_image(photo, self.max_size)
        if display_selection == 'origin':
            if self.origin_image_label is not None:
                self.origin_image_label.pack_forget()
                if self.makeup_image_label is not None:
                    self.makeup_image_label.pack_forget()
            self.origin_image_label = Label(image=photo)
            self.origin_image_label.image = photo
            self.origin_image_label.pack(side="left", padx=10, pady=10)
        elif display_selection == 'makeup':
            if self.makeup_image_label is not None:
                self.makeup_image_label.pack_forget()

            self.makeup_image_label = Label(image=photo)
            self.makeup_image_label.image = photo
            self.makeup_image_label.pack(side="right", padx=10, pady=10)
        print('Finish displaying image')


root = Tk()
root.title('Digital Makeup')
root.geometry("1200x700")
example_image_folder_path = '../assets'
app = DigitalMakeupApp(root, example_image_folder_path)
root.mainloop()
