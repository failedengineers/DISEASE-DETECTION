import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox,ttk
from tkinter import*
from PIL import Image,ImageTk



class gui:
    def __init__(self,root):
        self.root=root
        self.root.geometry("1350x740+0+0")
        self.root.title('FACE RECOGINATION ATTENDACE SYSTEM')
        self.model = self.load_pretrained_model()
        #image 1
        img0=Image.open(r"C:\python attendance\tk images\x.jpg")
        img0 = img0.resize((1350, 740), Image.Resampling.LANCZOS)
        self.photoimg0=ImageTk.PhotoImage(img0)
        flbl=Label(self.root,image=self.photoimg0)
        flbl.place(x=0,y=0,width=1350,height=740)
        text8=Button(self.root,text="UPLOAD FILE",font=("times new roman",10,"bold"),command=self.alltask,cursor="hand2",bg="darkblue",fg="white")
        text8.place(x=0,y=350,width=1350,height=60)
        
    def load_pretrained_model(self):
        model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(model.output)
        x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
        return tf.keras.models.Model(inputs=model.input, outputs=x)
    

    def preprocess_image(self, image_path):
        image = load_img(image_path, target_size=(224, 224))
        image = img_to_array(image)
        image = preprocess_input(image) 
        return np.expand_dims(image, axis=0)  

    def predict(self, image_path):
        image = self.preprocess_image(image_path)
        prediction = self.model.predict(image)[0][0]
        if prediction >0.5:
            messagebox.showinfo('DISEASE','DISEASE DETECTED',parent=self.root)
        else:
            messagebox.showinfo('DISEASE','NO DISEASE',parent=self.root)
            
    def uploadfile(self):
            while True:
                root =Tk()
                root.withdraw()  # Hide the main window
                self.file_path = filedialog.askopenfilename(title="Select a File")
                if self.file_path:
                    messagebox.showinfo('info','file uploaded succesfully',parent=self.root)
                    break
                else:
                    messagebox.showerror('error',' NO file uploaded succesfully',parent=self.root)
    def alltask(self):
        self.uploadfile()
        self.predict(self.file_path)
        
    



if __name__=="__main__":
    root=Tk()
    obj=gui(root)
    root.mainloop()
