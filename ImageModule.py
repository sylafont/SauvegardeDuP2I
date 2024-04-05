# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 21:04:22 2024

@author: sybil
"""
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np

def imageUploader(app):
    """Propose à l'utilisateur de récupérer une image de son ordinateur"""
    """L'image est redimensionnée"""
    fileTypes = [("Image files", "*.png;*.jpg;*.jpeg")]
    path = filedialog.askopenfilename(filetypes=fileTypes)
 
    if len(path):#Si l'utilisateur a bien sélectionné une image
        print("path : ", path)
        img = Image.open(path)
        width, height = img.size
        
        if(max(width, height) == width):
            height = round((height/width)*200)
            width = 200
        else :
            width = round((width/height)*200)
            height = 200
            
        img = img.resize((width, height))
        pixel_data = ProcessUserImageForNetworkClassification(img)
    
        pic = ImageTk.PhotoImage(img)
        
        app.geometry("560x320")
        label = Label(app, image=pic)
        label.config(image=pic)
        label.image = pic
        return True, label, pixel_data
 
    else:
        print("No file is chosen !! Please choose a file.")
        return False, None, None
        
def ProcessUserImageForNetworkClassification(img):
    img = img.resize((28, 28)) 
    img = img.convert('L')
    pixel_data = np.array(img.getdata())
    return np.divide(pixel_data,256)

def ProcessMnistImageForDisplay(image):
    image = image.reshape(28,28)
    img = Image.fromarray(image*256)
    img = img.resize((150, 150))
    photo = ImageTk.PhotoImage(img)
    label = Label(image=photo)
    label.image = photo
    return label

def LoadButtonWithImage(root, nameImage, height, width, color):
    """Renvoie un bouton qui a une image affichée dessus"""
    button = Button(root, bg = color)
    image = Image.open("Ressources/" + nameImage)
    image = image.resize((width, height))
    image = ImageTk.PhotoImage(image)
    button.config(image=image)
    button.image = image
    return button