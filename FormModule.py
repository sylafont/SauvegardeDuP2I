# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 16:48:33 2024

@author: sybil
"""
import tkinter 

class BoutonPlusMoins():
    def __init__(self, root, color) :
        self.root= root
        self.color = color
        self.Grandeboite = tkinter.Frame(self.root, bg= self.color)
        self.boite =tkinter.Frame(self.Grandeboite,pady = 10, bg = self.color)
        self.Grandeboite.pack()
        self.boite.pack()
        self.ButtonPlus = tkinter.Button(self.boite, text = "+" , command = self.AddOne, padx = 5)
        self.ButtonMoins = tkinter.Button(self.boite, text = "-" , command = self.SoustractOne, padx = 5)
        self.valueDisplayed = tkinter.Label(self.boite, text = "0", bg =self.color)
        self.listeButtonHiddenLayer = []
        self.ButtonMoins.pack(side = "left")
        self.ButtonPlus.pack(side = "left")
        self.valueDisplayed.pack(side = "left")
        self.AlreadyWarned = False
        
    
    def AddOne(self):
        counter = int(str(self.valueDisplayed["text"]))
        if counter < 4:
            counter += 1
            self.valueDisplayed.config(text=str(counter))
            self.valueDisplayed.pack()
            self.AddButton()
        elif self.AlreadyWarned == False:
            warning_couches = tkinter.Label(self.root, text = "Vous ne pouvez pas rentrer plus de 4 couches cachées ", bg = "pink")
            warning_couches.pack()
            self.AlreadyWarned = True
        
    
    def RetrieveButton(self):
        nbHiddenLayer = len(self.listeButtonHiddenLayer)
        if nbHiddenLayer!=0:
            self.listeButtonHiddenLayer[nbHiddenLayer-1].destroy()
            del self.listeButtonHiddenLayer[nbHiddenLayer-1]
    
    def AddButton(self):
        nbHiddenLayer = len(self.listeButtonHiddenLayer)
        Entry = NumericalEntry(self.Grandeboite, "Nombres de neurones sur la couche cachée n° " +str(nbHiddenLayer+1) , "int", self.color)
        self.listeButtonHiddenLayer.append(Entry)
    
    def SoustractOne(self) :
        counter = int(str(self.valueDisplayed["text"]))
        if counter !=0 :
            counter -= 1
            self.valueDisplayed.config(text=str(counter))
            self.valueDisplayed.pack(side = "bottom")
            self.RetrieveButton()      
    
    def IsItFilledIn(self):
        for EntryButton in self.listeButtonHiddenLayer:
            if not EntryButton.IsFilledIn():
                return False
        return True
    
    
    def GetEntriesValue(self):
        listeValues = [784]
        for EntryButton in self.listeButtonHiddenLayer:
            listeValues.append(EntryButton.GetNumber())
        listeValues.append(10)
        return listeValues
        
                        


class NumericalEntry():
    def __init__(self, root, labeltext, leType, color) :
        self.A_entry = tkinter.Entry(root, validate = 'key', validatecommand =(root.register(self.validate_input), '%P'))
        self.A_label = tkinter.Label(root, text=labeltext, bg =color)
        self.A_label.pack()
        self.A_entry.pack()
        self.type = leType
        
    def IsFilledIn(self):
        if not self.A_entry.get():
            return False
        return True
    
    def GetNumber(self):
        if not self.A_entry.get():
            return 0
        else :
            if self.type == "float":   
                return float(self.A_entry.get())
            else :
                return int(self.A_entry.get())
    
    def validate_input(self,value):
        if(self.type == "int") :
            if value.isdigit() or value =="":
                return True
        elif(self.type == "float"):
            try:
                IsFloat = float(value)
                return True
            except ValueError:
                if value == "":
                    return True  
                else:
                    return False
        else :
            raise ValueError("Le type d'input rentrée n'est pas valide")   
        return False
    
    def destroy(self):
        self.A_entry.destroy()
        self.A_label.destroy()
        

            
            