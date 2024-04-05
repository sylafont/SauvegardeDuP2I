import numpy as np
from AccesBaseDeDonnes import Load_Data_From_Pickle
from dependency import math, pickle
import matplotlib.pyplot as plt
from collections import Counter

class reseau() :
    """Réflechir à passer certaines variables en local"""
    def __init__(self, nb_epoch, liste_neurone_par_couche,lnr,  activation_type, loss_type, interface):
        self.lnr = lnr #Learning rate
        self.activation_type, self.loss_type = self.test_type(activation_type, loss_type)
        self.nb_batch = 10 #Nombre d'images par batch
        self.nb_epoch = nb_epoch 
        self.iteration_batch = 0 #compteur du nombre de batch déjà réalisé dans une époch, lors de l'entrainement
        self.nb_couches = len(liste_neurone_par_couche)
        self.liste_neurone_par_couche = liste_neurone_par_couche.copy()#Liste qui va contenir l'architecture du réseau : le nombre de neurones sur chaque couches
        self.initialisation_Matrice_Poids()
        self.train_image, self.train_label, self.image_test, self.label_test = Load_Data_From_Pickle(interface.db)#On rentre le nom de la base de données
        self.index_image_and_label = np.arange(60000)#inititialisation d'une liste qu'on va mélanger avant chaque epoch pour connaitre la répartition des images dans chaque batch (répartition aléatoire)
        #np.random.shuffle(self.index_image_and_label)
        self.train_nb_correct = [] #Pourcentage de prédictions correctes pour chaque epoch (sur le jeu d'entraînement)
        self.test_nb_correct = [] #Pourcentage de prédictions correctes pour chaque epoch (sur le jeu de test)
        self.predic_set = [] #Liste qui va prendre toutes les prédictions du réseau après une session de test des résultats sur le jeu de test ou d'entrainement
        self.ind_echec = []
        self.interface = interface #Il est nécessaire d'avoir un attribut interface de la classe interface utilisateur
        #pour que le réseau puisse communiquer à l'interface sur le déroulé de l'entraînement et ainsi permettre l'affichage de la barre de progression

    def test_type(self, activation, loss) :
        """Fonction qui vérifie que l'utilisateur a rentré une chaine de caractère valide pour la fonction d'activation et de coût"""
        
        if activation != "sigmoid" or loss != "least_squares" :
            raise ValueError("Le type d'activation ou de fonction de cout rentrée n'est pas valide")
        else :
            return activation, loss
        
    
    def Index_Images_For_New_Batch(self ) :
        """Fonction qui va récupérer les images qui doivent être ajoutées au nouveau batch d'entraînement"""
        self.label_batch = []
        nb_neurone_premiere_couche = self.liste_neurone_par_couche[0]
        images = np.ones((nb_neurone_premiere_couche, self.nb_batch ))
        s=0
        
        for i in range(self.iteration_batch*self.nb_batch, self.iteration_batch*self.nb_batch + self.nb_batch) : 
            index = self.index_image_and_label[i]
            image = self.train_image[index, :]
            label = self.train_label[index]
            images[:,s] = image
            s= s+1
            self.label_batch.append(label)
            
        return images
    
    def Feed_InputLayer(self, images) :
        self.a = [] #Initialisation d'une liste contenant les activations du réseau pour chaque couche
        self.z = [] #Initialisation d'une liste contenant les sorties brutes des neurones (la somme pondéré de toutes les activations des neurones précédents)
        self.a.append(images)

    """def Test_Input_PerBatch(self):
        for i in range(self.nb_batch):
            print("label : ", self.label_batch[i])
            plt.figure()
            plt.imshow(self.a[0][:,i].reshape([28,28]), cmap='gray')
            plt.show()"""

    def activation(self, num):
        """Fonction à compléter au fur et à mesure que d'autres fonctions d'activations sont rajoutées"""
        
        if self.activation_type == "sigmoid" :
            return 1/(1 + math.exp(-num))
    
    def derive_Activation(self):
        """Fonction qui génère les dérivées de toutes les activations du réseau"""
        if self.activation_type == "sigmoid" :
            self.derive_Z = []
            for activation in self.a:
                dZ = activation*(1 - activation) #La fonction d'activation étant la sigmoid sa dérivée est a*(1-a) avec a la fonction sigmoid
                self.derive_Z.append(dZ)
        else :
            raise ValueError("La fonction d'activation rentrée n'est pas valide")

    def derive_Cout(self, activation_ouput_layer):
        if self.loss_type == "least_squares" :
            self.derive_cout_activation = (activation_ouput_layer - self.y_real)
        else :
            raise ValueError("La fonction d'activation rentrée n'est pas valide")

    def initialisation_Matrice_Poids(self):
        """Fonction qui initialise les matrices de poids et les biais avec des valeurs aléatoires"""
        self.w =[]
        self.b =[]
        
        for j in range(0, self.nb_couches- 1): #On parcourt tout les couches du réseau en commencant par la couche d'entrée
            
            nb_neurone_couche_suivante = self.liste_neurone_par_couche[j+1]#On récupère le nombre de neurones de la couche suivante
            nb_neurone_cette_couche = self.liste_neurone_par_couche[j] #On récupère le nombre de neurones pour cette couche
            b0 = np.random.uniform(low=-1.0, high=1.0, size=(nb_neurone_couche_suivante,1)) #on commence à j+1 car les neurones de la couche d'entrée n'ont pas de biais
            w0 = np.ones((nb_neurone_couche_suivante, nb_neurone_cette_couche))
            self.b.append(b0)
            for i in range(0, nb_neurone_couche_suivante ) : 
                w0[i,:]= np.random.uniform(low=-1.0, high=1.0, size=nb_neurone_cette_couche)
            self.w.append(w0)

    """"def feedInput(self) :
    
        nb_neurone_premiere_couche = self.liste_neurone_par_couche[0]
        a0 = np.ones((nb_neurone_premiere_couche, self.nb_batch ))
        for i in range(self.nb_batch) :    #feed input neurone
            image, label = Get_image_And_Label(i)
            a0[: ,i] = image
        self.a.append(a0)"""


    def forward(self, input) :
        """Fonction qui propage les inputs vers l'avant en calculant les activations du réseau"""
        
        self.Feed_InputLayer(input)

        for i in range(self.nb_couches-1) :
            new_z = (np.matmul(self.w[i], self.a[i]) + self.b[i])
        
            self.z.append(new_z)
            activations = np.vectorize(self.activation)
        
            new_a = activations(new_z)
            self.a.append(new_a)
        
        activation_output_layer = self.a[self.nb_couches -1]
        return activation_output_layer

    def backward(self, activation_ouput_layer):
        """Algorithme de rétro-propagation"""
        self.derive_Activation()
        self.derive_Cout(activation_ouput_layer)
        self.Compute_delta()
        self.Change_Biais_and_Weights()

    def Transform_Label_To_Vector(self) :
        """Fonction qui permet de convertir le label en un vecteur de la forme [0 0 0 0 1 0...0] pour comparer avec l'activation finale du réseau"""
        nb_output_neurones = self.liste_neurone_par_couche[self.nb_couches - 1]
        
        self.y_real = np.zeros((nb_output_neurones, self.nb_batch))

        for i in range(self.nb_batch):
            index_label = int(self.label_batch[i])
            self.y_real[index_label, i ] = 1
    
    def Compute_Loss_Function(self): 
        """Calcule de la fonction de cout destinée uniquement à l'affichage dans la console"""
        y_predicted = self.a[self.nb_couches-1]

        if self.loss_type == "least_squares" :
            diff_matrices = (self.y_real - y_predicted)**2
            self.cost = np.sum(diff_matrices, axis =0)/2
            return np.mean(self.cost)
        else :
            raise ValueError("La fonction de cout rentrée n'est pas valide")
        print("Cout de la fonction : ", np.mean(self.cost))

    def Compute_delta(self):
        """Etape importante pour l'algorithme de rétro-propagaton"""
        """Les deltas sont les gradients des biais par rapport à la fonction de coût"""
        """Ils permettent ensuite de calculer les poids et les biais des couches précédentes"""
        self.deltas = [0]*self.nb_couches
        derive_activation_outputlayer = self.derive_Z[self.nb_couches-1]
        delta_output_layer = derive_activation_outputlayer * self.derive_cout_activation
        self.deltas[self.nb_couches - 1] = delta_output_layer

        for i in range(self.nb_couches-2, 0, -1): #On parcourt les couches du réseaux en commencant les dernières
            delta = np.matmul(self.w[i].T,self.deltas[i+1])*self.derive_Z[i]
            self.deltas[i] = delta

    def Change_Biais_and_Weights(self):
        """Descente de gradient"""
        for i in range(self.nb_couches-1) :
            gradient_bais = np.mean(self.deltas[i+1], axis =1, keepdims= True)#On fait la moyenne des 10 gradients calculés pour chacune des images du batch
            self.b[i] = self.b[i] - self.lnr*gradient_bais #Ajustement des valeurs des biais
            new_shape_a = np.reshape(self.a[i], (1, np.shape(self.a[i])[0], np.shape(self.a[i])[1]))
            new_shape_delta = np.reshape(self.deltas[i+1], (np.shape(self.deltas[i+1])[0],1, np.shape(self.deltas[i+1])[1]))
            gradient_weight = np.mean(new_shape_delta*new_shape_a, axis=2)
            self.w[i] = self.w[i] - self.lnr*gradient_weight #Ajustement des valeurs des poids 
        

    
    def Train_network(self):
       
        for j in range(self.nb_epoch) :
            print("Epoch : ", j+1)
            np.random.shuffle(self.index_image_and_label) #On change la composition des batchs entre chaque epoch pour ne pas que les images soient présentées toujours dans le même ordre
            self.costs = np.zeros((6000))
            self.iteration_batch = 0
            for i in range(6000) : #Avec un jeu de test de 60 000 images et 10 images par batch, il y a 6000 itérations par epoch
                images = self.Index_Images_For_New_Batch()#On récupère les images pour le nouveau batch
                self.Transform_Label_To_Vector() #On convertit les labels en vecteur [0 0 1 ...0 0]
                activation_ouput_layer = self.forward(images)
                self.backward(activation_ouput_layer)
                self.costs[i] = self.Compute_Loss_Function()
                self.iteration_batch=self.iteration_batch+1
                if (j*6000 + (i+1))%(600*self.nb_epoch) == 0: #Dix fois par entraînement la bar de progression se réactualise
                    self.interface.DisplayProgressionBar(j*6000 + (i+1),600*self.nb_epoch)
                    self.Test_Network(self.train_image.T, self.train_nb_correct, self.train_label)#On récupère la proportion d'images correctement prédites sur le jeu d'entraînement
                    self.Test_Network(self.image_test.T, self.test_nb_correct, self.label_test)#On récupère la proportion d'images correctement prédites sur le jeu de test
            print("Cout de la fonction : ", np.mean(self.costs))
        self.ComputeIndEchec()#On récupère les indices des images qui ont mal été labelisés, car ces informations serviront pour les pages "Image mal classées" et "Statistiques" 

    def Test_Network(self, array_image, liste_score_par_epochs, labels) :
        """Cette fonction est appliquée soit sur le jeu d'entraînement soit sur le jeu de test"""
        """Elle permet de récupérer les information pour le graphique de l'évolution du score en fonction du niveau d'entraînement"""
        activation_ouput_layer = self.forward(array_image)
        self.predic_set = np.argmax(activation_ouput_layer, axis =0)#On récupère l'indice de la valeur maximale du vecteur. L'indice est égale au chiffre prédit par le réseau
        nb_correct = self.predic_set == labels #On crée une array composée de True ou False en fonction de si l 'image à cette indice a bien été reconnue ou non
        pourcentage_correct = ((np.sum(nb_correct))/len(nb_correct))*100
        liste_score_par_epochs.append(pourcentage_correct)
        print("Le taux de prédiction correcte est de : ", pourcentage_correct, "%")
    
    
    def ComputeIndEchec(self) :
        prediction = self.a[self.nb_couches-1]
        self.prediction = np.argmax(prediction, axis =0)
        self.ind_echec = np.where(self.prediction != self.label_test)
        
    def Retrieve_Error_After_Training(self) : 
        labelNumbers =[]
        occurences =[]
        for i in range(10):
            falsePredictionForiNumber = np.extract(self.label_test[self.ind_echec] == i, self.prediction[self.ind_echec])
            counter = Counter(falsePredictionForiNumber) #Renvoie un dictionnaire avec les occurences de chaque valeur
            number = list(counter.keys())
            occurence = list(counter.values())
            labelNumbers.append(number)
            occurences.append(occurence)
        return labelNumbers, occurences
        
        """for i in range(20) :
            print("Vrai chiffre : ", label_echec[i])
            print("Prediction : ", prevision_echec[i])
        #joblib.dump(label_echec, "label_echec.pkl")"""
     
    def ToString(self):
        """Fonction destinée à afficher les informations spécifiques au réseau"""
        text = "Réseau composé de " + str(self.nb_couches) +" couches \n dont "
        for i in range(self.nb_couches):
            if i==0:
                numeroCouche = "d'entrée"
            elif i==self.nb_couches-1:
                numeroCouche = "de sortie"
            else : 
                numeroCouche = str(i+1)   
            text = text + str(self.liste_neurone_par_couche[i]) + " neurones sur la couche " + numeroCouche + "  \n"
        text = text + " \n Entrainé avec la base de donnée MNIST, composée de 10 000 images test et 60 000 images d'entrainement \n"
        text = text + "La fonction d'activation des neurones est la "+ self.activation_type + "\n La fonction de coût utilisée est la " + self.loss_type
        text = text + "\n" + "Le réseau est entrainé par batch de " + str(self.nb_batch)+ " images"
        text = text + "\n" + "Son score de précision est de " + str(round(self.test_nb_correct[-1],2))
        text = text + "\n" + "La proportion d'images mal classées est de " + str(len(np.squeeze(self.ind_echec))) + " sur 10 000"
        return text
        
    
    def Save_Network(self, fileName) :
        del self.interface
        try :
            with open("Networks/"+fileName +".pickle", "wb") as f :
                pickle.dump(self,f,protocol=pickle.HIGHEST_PROTOCOL)
                print("Ca devrait avoir réussit")
        except Exception as ex :
            print("nameFile : ", fileName)
            print("Error during pickling object (Possibly unsuppoted) : ", ex)
            return False
        return True
    
    @classmethod
    def Load_Network(cls, fileName ) :
        try :
            with open("Networks/"+fileName + ".pickle", "rb") as f :
                return pickle.load(f)
        except Exception as ex :
            print("Error during unpickling object (Possibly unsupported) : ", ex)

   

#network = reseau(10, [784,30, 10] , "sigmoid", "least_squares")
#network.Train_network()
#network.forward()
#print("Activation output layer : ")
#print(network.a[2])

#DecisionSauvegarde = input("Voulez-vous sauvegarder ce réseau ? O/N")

#if (DecisionSauvegarde == 'O'):
    #network.Save_Network()