# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 16:01:50 2024

@author: sybil
"""
import tkinter
from tkinter import ttk
import matplotlib.pyplot as plt
from FormModule import BoutonPlusMoins, NumericalEntry
from functools import partial
from reseau import reseau
from PlotStatModule import plotGraph, plotPie
import numpy as np
from ImageModule import imageUploader, ProcessMnistImageForDisplay, LoadButtonWithImage
import os
from AccesBaseDeDonnes import Save_Data_Base, IsInFolder


class Interface(): 
    def __init__(self, db) :
        self.db = db #Nom de la base de donnée
        self.fenetre = tkinter.Tk()
        self.fenetre.geometry("400x500") #On empêche la fenêtre de se redimensionner pour être sur que les widgets se placent bien
        self.fenetre.resizable(False, False)
        self.reseauASauvegarder = False
        self.bg_color = '#F9DDAD'
        self.Page_Accueil()
        
    def Page_Accueil(self):
        self.clearPreviousPage(400,500,0,)
        titre = tkinter.Label(self.fenetre, text = "Accueil", bg = "orange")
        
        if not hasattr(self, "network"): #Si l'utilisateur n'as pas encore choisit d'entraîner ou de charger un réseau
            choix1 = LoadButtonWithImage(self.fenetre, "BoutonEntrainement.png", 132, 262 , self.bg_color)
            choix1.config(command =  self.NetworkSpecificationPage)#Renvoie vers la page formulaire pour entraîner le réseau
            if len(self.GetPreviousNetworks())!=0 : #S'il n'y a au moins un reseau sauvegardé dans le dossier Networks
                choix2 = LoadButtonWithImage(self.fenetre,"BoutonLoad.png", 132, 262, self.bg_color)#On propose à l'utilisateur de pouvoir charger un réseau
                choix2.config(command =self.ChoseNetworkPopUp)
                choix2.grid(row = 2, column=0, sticky = "n", pady=(0,10))
        else : #L'utilisateur a déjà chargé ou entraîné un réseau
            choix1 =LoadButtonWithImage(self.fenetre, "BoutonFonctionnalite.png", 132, 262, self.bg_color)
            choix1.config(command = self.MenuStatPage)
            choix2 = LoadButtonWithImage(self.fenetre, "BoutonEntrainement.png", 132, 262, self.bg_color)
            choix2.config(command =  self.CheckIfUserWantToProceed)
            choix2.grid(row = 2, column=0, sticky = "n", pady=(0,10))
            if len(self.GetPreviousNetworks())!=0 :
                choix3 = LoadButtonWithImage(self.fenetre, "BoutonLoad.png", 132, 262, self.bg_color)
                choix3.config(command =self.ChoseNetworkPopUp)
                choix3.grid(row = 3, column=0, sticky = "n")
        
        self.fenetre.columnconfigure(0, weight =1)
        titre.grid(row = 0, column=0, sticky = "n", pady=10 )
        choix1.grid(row = 1, column=0, sticky = "n", pady=(0,10))
        
    def DisplayProgressionBar(self, nb_epoch, nb_total):
        """Fonction appelée par le réseau au cours de l'entraînement"""
        progression = round((nb_epoch/nb_total)*10)
        self.pb['value'] =  progression #La valeur de la barre de progression est mise à jour
        label = tkinter.Label(self.barPlusInfo, text = str(progression)+ "%", bg = self.bg_color)
        label.grid(row=1)
        self.fenetre.update_idletasks() #On force la fenêtre à s'actualiser sinon les calculs du réseau seront fait avant l'actualisation de la barre de progression
        
    def CheckIfUserWantToProceed(self):
        """Fonction qui prévient l'utilisateur qu'il a un réseau non sauvegardé et qu'il risque de le perdre en poursuivant son action"""
        proceed = True
        if self.reseauASauvegarder == True :
            msg = "Vous avez un réseau non sauvegardé, si vous tentez d'en entraîner un autre ces données seront perdues"
            proceed = tkinter.messagebox.askokcancel(title=None, message=msg)
        if proceed  == True :
            self.NetworkSpecificationPage()

    def GetPreviousNetworks(self):
        """Fonction qui récupère tous les fichiers dans le dossier Networks, et compile les noms dans une liste"""
        for path, subdirs, files in os.walk("Networks"):
            networkName=[]
            for name in files:
                if ".pickle" in name :
                    networkName.append(name.replace(".pickle", ""))
        return networkName
            
    def ChoseNetworkPopUp(self):
        """Fonction qui fait apparaitre une popUp pour demander à l'utilisateur quel réseau il souhaite charger"""
        self.fenetre.attributes('-disabled', True) #On empêche l'utilisateur de pouvoir interargir avec la fenêre tant que la popUp est là
        popUp = tkinter.Toplevel(self.fenetre, bg = self.bg_color)
        popUp.geometry("250x200")
        popUp.resizable(False, False)
        popUp.protocol("WM_DELETE_WINDOW", partial(self.DestroyPopUpAndEnableWindow, popUp))#Quand l'utilisateur appuie sur la croix rouge de la popUp cela réactive la fenêtre en plus de détruire la popup
        title = "Choississez le réseau à charger"
        titleLabel = tkinter.Label(popUp, text = title, bg="orange")
        titleLabel.pack(side ='top', pady=20)
        networkName = self.GetPreviousNetworks()
        tkVarS = tkinter.StringVar(popUp)
        tkVarS.set(networkName[0])
        userEntry = tkinter.OptionMenu(popUp, tkVarS, *networkName) #Création du bouton à choix multiple
        userEntry.pack(expand = True)
        submit = tkinter.Button(popUp, text = "Valider", command = partial(self.CheckLoadedNetworkAndValidate,popUp, tkVarS))
        submit.pack(side = "right",  padx = 30, pady= (0,50))
        annuler = tkinter.Button(popUp, text = "Annuler", command = partial(self.DestroyPopUpAndEnableWindow, popUp))
        annuler.pack(side = 'left', padx = 30, pady= (0,50))
        self.fenetre.wait_window(popUp)
            
    def CheckLoadedNetworkAndValidate(self, popUp, fileName):
        """Chargement du réseau après que l'utilisateur ait rentré son choix"""
        self.network = reseau.Load_Network(fileName.get())
        self.network.interface = self
        if self.network != None:
            self.fenetre.attributes('-disabled', True)
            popUp.destroy()
            self.MenuStatPage()
        else :
            msg = "Le réseau n'a pas pu être chargé, veuillez en choisir un autre"
            tkinter.messagebox.showerror(title=None, message=msg)
        
    
    def FenetrePopUpSauvegarde(self):
        """PopUp qui apparait quand l'utilisateur clique sur le bouton sauvegarde"""
        self.fenetre.attributes('-disabled', True)
        popUp = tkinter.Toplevel(self.fenetre)
        popUp.protocol("WM_DELETE_WINDOW", partial(self.DestroyPopUpAndEnableWindow, popUp))
        popUp.geometry("280x150")
        popUp.resizable(False, False)
        popUp.title = "Sauvegarde du réseau"
        message = "Veuillez entrer le nom du fichier à sauvegarder"
        messageLabel = tkinter.Label(popUp, text = message, bg = "orange")
        entryNameFile =  tkinter.Entry(popUp)
        annulationButton = tkinter.Button(popUp, text = "Annuler", command = partial(self.DestroyPopUpAndEnableWindow, popUp))
        validationButton = tkinter.Button(popUp, text = "Valider", command = partial(self.SaveAndFeedback, popUp, entryNameFile ))
        messageLabel.grid(row=0,column=0, columnspan=2,padx= 15)
        entryNameFile.grid(row=1, column=0, columnspan=2, pady= 15 )
        annulationButton.grid(row =2, column=0, pady=15)
        validationButton.grid(row =2, column=1, pady = 15)
        
    def SaveAndFeedback(self, popUp, nameFile):
        reussite =False
        if len(nameFile.get())  != 0: #On s'assure que l'utilisateur a rentré un nom, sinon on refuse la sauvegarde
            reussite = self.network.Save_Network(nameFile.get()) #On vérifie qu'il n'y a pas eu de problème lors de la sauvegarde du réseau
        
        if reussite == True:
            self.fenetre.attributes('-disabled', False)
            self.reseauASauvegarder=False
            popUp.destroy()
            tkinter.messagebox.showinfo(title=None, message="Réseau bien sauvegardé !")
            self.MenuStatPage()
            
        else :
            msg ="Une erreur est survenue lors de la sauvegarde du réseau \n Veuillez essayer un autre nom pour le fichier"
            tkinter.messagebox.showerror(title=None, message=msg)
            
    def DestroyPopUpAndEnableWindow(self, popUp):
        self.fenetre.attributes('-disabled', False)
        popUp.destroy()
            
        
    def clearPreviousPage(self, width, hight, n_column):
        for child in self.fenetre.winfo_children() :
            child.grid_remove()
            child.destroy()
            
        self.fenetre.destroy()
        self.fenetre = tkinter.Tk()
        self.fenetre.configure(bg=self.bg_color)
        self.fenetre.resizable(False, False)
        self.fenetre.geometry(str(width)+"x"+str(hight))
        
        if self.reseauASauvegarder ==True : #Si l'utilisateur a sn=on réseau non sauvegardé on affiche le bouton de sauvegarde en haut à droite de la chaque page
            buttonSauvegarde = LoadButtonWithImage(self.fenetre, "iconeSauvegarde.png", 37,50, self.bg_color)
            buttonSauvegarde.config(command = self.FenetrePopUpSauvegarde)
            buttonSauvegarde.grid(row = 0, column = n_column, sticky = "e", padx=30)
            
    def NetworkSpecificationPage(self):
        self.reseauASauvegarder= False
        self.clearPreviousPage(400,540,0)
        compteurClick = tkinter.IntVar()
        compteurClick.set(0)
        CompteurHiddenLayer =0
        titre = tkinter.Label(self.fenetre, text = "Architecture du réseau : ", bg = "orange")
        questionHiddenLayer = tkinter.Label(self.fenetre, text = "Combien de couches cachées souhaitez vous mettre dans le réseau ? ", bg = self.bg_color)
        returnButton = tkinter.Button(self.fenetre, text = "Retour", command =   self.Page_Accueil)
        titre.pack()
        questionHiddenLayer.pack()
        ButtonLayer = BoutonPlusMoins(self.fenetre, self.bg_color)
        EpochNumber = NumericalEntry(self.fenetre, "Combien d'épochs voulez vous réaliser?", "int", self.bg_color)
        LearningRate = NumericalEntry(self.fenetre, "Définissez un learning rate", "float", self.bg_color)
        ValidateButton = tkinter.Button(self.fenetre, text = "Valider", command = partial(self.CheckValidation, EpochNumber,ButtonLayer,LearningRate, compteurClick ))
        returnButton.pack()
        ValidateButton.pack()
        
    def CheckValidation(self, Epoch, Layers, Lnr, compteur) :
        compteur.set(compteur.get() + 1)
        if Epoch.IsFilledIn() and  Layers.IsItFilledIn() and Lnr.IsFilledIn():
            ValuesHiddenLayer = Layers.GetEntriesValue()
            self.network = reseau(Epoch.GetNumber(),ValuesHiddenLayer,Lnr.GetNumber(), 'sigmoid', 'least_squares', self)#Pour l'instant les fonctions d'activations et de cout ne sont pas modifiable
            self.NewNetworkLoadingPage()
            self.reseauASauvegarder = True
            self.SucessTrainingPage()
        else :
            if compteur.get() ==1:
                warning = tkinter.Label(self.fenetre, text = "Vous devez remplir tous les champs avant de valider ", bg = "pink")
                warning.pack()
    
    
    def NewNetworkLoadingPage(self):
        
        notification = tkinter.Label(self.fenetre, text = "Le réseau est en cours d'entrainement...\nCela peut prendre plusieurs minutes", bg="orange")
        notification.pack(pady=10)
        self.barPlusInfo = tkinter.Frame(self.fenetre, bg =self.bg_color)
        self.barPlusInfo.pack(pady =(15,0))
        self.pb = ttk.Progressbar(self.barPlusInfo,orient='horizontal',mode='determinate',length=100)
        self.pb['value'] =  0
        label = tkinter.Label(self.barPlusInfo, text = "0%", bg = self.bg_color)
        label.grid(row=1)
        self.pb.grid(row=0)
        self.fenetre.update_idletasks()
        self.fenetre.attributes('-disabled', True)
        self.network.Train_network()
        self.fenetre.attributes('-disabled', False)
        
    def SucessTrainingPage(self):
        msg = "Entrainement réussi ! \nVotre réseau a reconnu "+ str(round(self.network.test_nb_correct[-1],2))+"% des images du jeu de test"
        tkinter.messagebox.showinfo(title="Entrainement", message=msg)
        self.Page_Accueil()
            
        
    def MenuStatPage(self):
        self.clearPreviousPage(400,520,0)
        titre = tkinter.Label(self.fenetre, text = "Fonctionnalités du réseau", bg = "orange")
        choix1 = LoadButtonWithImage(self.fenetre, "BoutonReconnaissance.png", 132, 262, self.bg_color)
        choix1.config(command = self.Entrer_Image)
        choix2 = LoadButtonWithImage(self.fenetre, "BoutonImageErreur.png", 132, 262, self.bg_color)
        choix2.config(command = self.Exemple_Images)
        choix3 =  LoadButtonWithImage(self.fenetre, "BoutonStat.png", 132, 262, self.bg_color)
        choix3.config(command =self.Statistiques_du_Reseau)
        returnButton = tkinter.Button(self.fenetre, text = "Retour", command =  self.Page_Accueil)
        self.fenetre.columnconfigure(0, weight =1)
        titre.grid(row = 0, column=0, sticky = "n", pady=10 )
        choix1.grid(row = 1, column=0, sticky = "n", pady=(0,10) )
        choix2.grid(row = 2, column=0, sticky = "n", pady=(0,10) )
        choix3.grid(row = 3, column=0, sticky = "n", pady=(0,10) )
        returnButton.grid(row = 4, column=0, sticky = "n" )

    def Entrer_Image(self):
        self.clearPreviousPage(450,450,0)
        titre = tkinter.Label(self.fenetre, text = "Soumettre une image", bg = "orange")
        self.fenetre.columnconfigure(0, weight =1)
        titre.grid(row=0, column=0)
        interaction_utilisateur, label, pixel_data = imageUploader(self.fenetre)
        if interaction_utilisateur == True:
            prediction =  np.argmax(self.network.forward(np.expand_dims(pixel_data, 1)))
            print("prediction shape : ", np.shape(self.network.forward(np.expand_dims(pixel_data, 1))))
            labelPrediction = tkinter.Label(self.fenetre, text = "Prediction du réseau: "+ str(prediction), bg = "orange")
            label.grid(row=1,column=0)
            labelPrediction.grid(row=2, column=0)
        Button_Upload = tkinter.Button(self.fenetre, text = "Tester une image", command = self.Entrer_Image)
        Button_Upload.grid(row=3,column=0)
        returnButton = tkinter.Button(self.fenetre, text = "Retour", command =   self.MenuStatPage)
        returnButton.grid(row=4,column=0)
        
    def Exemple_Images(self):
        self.clearPreviousPage(1200,600,2)
        self.fenetre.grid_columnconfigure(1, weight=1)
        titre = tkinter.Label(self.fenetre, text = "Images mal classées", bg = '#FA8072')
        titre.grid(row=0, column=1, sticky="s")
        self.fenetre.grid_rowconfigure(0, weight=1)
        self.Place9Images()
        returnButton = tkinter.Button(self.fenetre, text = "Retour", command =  self.MenuStatPage)
        returnButton.grid(row=4, column=1, sticky="e", padx= 110)
        refreshButton = tkinter.Button(self.fenetre, text = "Voir d'autres exemples", command =  self.Exemple_Images)
        refreshButton.grid(row=4, column=1, sticky="w", padx=70)
        self.fenetre.grid_rowconfigure(4, weight=1)

    def Statistiques_du_Reseau(self):
        self.clearPreviousPage(1030,670,2)
        titre = tkinter.Label(self.fenetre, text = "Données sur le réseau", bg = "orange")
        InfoReseau = tkinter.Label(self.fenetre, text = self.network.ToString(), background="#FCEFDF")
        titre.grid(row=0, column =0, columnspan =2)
        InfoReseau.grid(row=1, column =0, columnspan =2, pady=10)
        y = [self.network.train_nb_correct,self.network.test_nb_correct]
        fig, graph = plotGraph(self.fenetre, "Pourcentage de réussite durant l'entraînement", np.swapaxes(y,0,1), ['Training Set', 'Testing set'], self.bg_color)
        graph.grid(row =2,column =0, padx = 20, sticky = "nsew")
        self.ManagePie(self.fenetre)
        returnButton = tkinter.Button(self.fenetre, text = "Retour", command =  self.MenuStatPage)
        returnButton.grid(row=3, column=0, columnspan = 2)
        
    
    def ManagePie(self, master):
        pieIndex = tkinter.IntVar()
        pieIndex.set(0)
        conteneur = tkinter.Frame(master, bg = self.bg_color)
        conteneur.grid(row = 2, column=1, sticky = "nsew")
        labels, occurences = self.network.Retrieve_Error_After_Training()
        buttonLeft = LoadButtonWithImage(conteneur,"BoutonFlecheGauche.png",50,50, self.bg_color)
        buttonRight = LoadButtonWithImage(conteneur,"BoutonFlecheDroite.png",50,50, self.bg_color)
        buttonLeft.config(command = partial(self.setPie, conteneur, pieIndex,"gauche",labels, occurences))
        buttonRight.config(command = partial(self.setPie, conteneur, pieIndex,"droite",labels, occurences))
        buttonLeft.pack(side="left")
        buttonRight.pack(side="right")
        self.DisplayPie(conteneur,pieIndex, occurences, labels)
       
        
    def setPie(self,master, index, direction, labels, occurences):
        if (direction == "droite") :
            if index.get() !=9:
                index.set(index.get()+1)
            else :
                index.set(0)
        elif(direction == "gauche"):
            if index.get() !=0:
                index.set(index.get()-1)
            else :
                index.set(9)
        else : 
            raise ValueError("Cette direction n'existe pas!")
            
        plt.close(self.fig)
        self.actualPie.destroy()
        self.detailPie.destroy()
        self.DisplayPie(master, index, occurences, labels )
        
        
    def DisplayPie(self,master, index, occurences, labels ):
        nb_false_prediction = len(np.squeeze(self.network.ind_echec))
        nb_false_prediction_for_index = sum(occurences[index.get()])
        pourcentageError =  round(nb_false_prediction_for_index/nb_false_prediction *100, 2)
        titre = "Répartition des erreurs pour le "+ str(index.get()) 
        DetailedStat = "Les erreurs sur le chiffre " + str(index.get())+ " représentent " +str(pourcentageError)+ "% de l'erreur totale \n soit " + str(nb_false_prediction_for_index) +" fausses prédictions sur "+ str(nb_false_prediction)
        self.detailPie = tkinter.Label(master, text = DetailedStat, bg= "orange")
        self.fig, self.actualPie = plotPie(master,titre, labels[index.get()], occurences[index.get()],self.bg_color)  
        self.actualPie.pack(expand=1)
        self.detailPie.pack(side = "bottom")
        
    
    def Place9Images(self):
        ind_echec = np.squeeze(self.network.ind_echec)
        indexImage = np.random.randint(len(ind_echec), size=6)
        for i in range(6):
            indice = ind_echec[indexImage[i]]
            ChiffrePredit = self.network.prediction[indice]
            VraiChiffre = self.network.label_test[indice]
            LabelPredict = tkinter.Label(self.fenetre, text='Prédiction : '+ str(ChiffrePredit),  bg='#CD5C5C')
            VraiLabel =  tkinter.Label(self.fenetre, text='Label : '+ str(VraiChiffre), bg='#FFA07A')
            image = self.network.image_test[indice].copy()
            self.fenetre.grid_rowconfigure(int(i/3)+1, weight=3)
            self.fenetre.grid_columnconfigure(i%3, weight=1)
            LabelImage = ProcessMnistImageForDisplay(image)
            LabelImage.grid(row=int(i/3)+1, column=i%3, sticky="s", pady=20)
            LabelPredict.grid(row=int(i/3)+1, column=i%3, sticky="nw", pady=50, padx=70)
            VraiLabel.grid(row=int(i/3)+1, column=i%3, sticky="ne", pady=50, padx=70)
            
db = "mnist_data_base"
if not IsInFolder(db, "DataBase"):
    Save_Data_Base(db)

I1 = Interface(db)
I1.fenetre.mainloop()