import numpy as np
from AccesBaseDeDonnes import Get_image_And_Label, Load_Data_From_Pickle
from dependency import math, pickle
import matplotlib.pyplot as plt
from collections import Counter

class reseau() :
    """Réflechir à passer certaines variables en local"""
    def __init__(self, nb_epoch, liste_neurone_par_couche,lnr,  activation_type, loss_type, interface):
        
        self.lnr = lnr #Learning rate
        self.activation_type, self.loss_type = self.test_type(activation_type, loss_type)
        self.nb_batch = 10
        self.nb_epoch = nb_epoch
        self.iteration_batch = 0
        self.nb_couches = len(liste_neurone_par_couche)
        self.liste_neurone_par_couche = liste_neurone_par_couche.copy()
        self.initialisation_Matrice_Poids()
        self.data_image_pool, self.label_pool, self.image_test, self.label_test = Load_Data_From_Pickle()
        self.index_image_and_label = np.arange(60000)#intitialisation d'une liste qu'on va shuffle
        np.random.shuffle(self.index_image_and_label)#peut être à changer de place
        #Mettre un self.index_mauvaise_predic
        self.train_nb_correct = [] #Pourcentage de prédiction correcte pour chaque epochs (sur le training set)
        self.test_nb_correct = [] ##Pourcentage de prédiction correcte pour chaque epochs (sur le test set)
        self.predic_test_set = []
        self.ind_echec = []
        #toutes les images et labels de la base de données MNIST dont on enlève progressivement certaines quand elles ont été vues
        #remis à jour à chaque epochs
        self.interface = interface

    def test_type(self, activation, loss) :
        """Fonction qui vérifie que l'utilisateur a rentré une chaine de caractère valide pour la fonction d'activation et de cout"""
        
        if activation != "sigmoid" or loss != "least_squares" :
            raise ValueError("Le type d'activation ou de fonction de cout rentrée n'est pas valide")
        else :
            return activation, loss
        
    
    def Index_Images_For_New_Batch(self) :
        #self.a = []
        #self.z = []
        self.label_batch = []
    
        #Le nombre n dépend du nombre d'image par batch
        nb_neurone_premiere_couche = self.liste_neurone_par_couche[0]
        #a0 = np.ones((nb_neurone_premiere_couche, self.nb_batch ))
        images = np.ones((nb_neurone_premiere_couche, self.nb_batch ))
        s=0
        
        for i in range(self.iteration_batch*self.nb_batch, self.iteration_batch*self.nb_batch + self.nb_batch) :#a voir si je peux pas plutot del les index des images déjà utilisées en partant du dernier item de la liste
            index = self.index_image_and_label[i]
            image = self.data_image_pool[index, :]
            label = self.label_pool[index]
            images[:,s] = image
            #a0[: ,s] = image
            s= s+1
            self.label_batch.append(label)
            
        #self.Feed_InputLayer(images)
        return images
        #print("shape label : ", np.shape(label))
        #self.a.append(a0)
    
    def Feed_InputLayer(self, images) :
        self.a = []
        self.z = []
        self.a.append(images)

    def Test_Input_PerBatch(self):
        for i in range(self.nb_batch):
            print("label : ", self.label_batch[i])
            plt.figure()
            plt.imshow(self.a[0][:,i].reshape([28,28]), cmap='gray')
            plt.show()

    def activation(self, num):
        """Méthode à compléter au fur et à msure qu'on teste des fonction d'activation différente"""
        
        if self.activation_type == "sigmoid" :
            return 1/(1 + math.exp(-num))
    
    def derive_Activation(self):
        if self.activation_type == "sigmoid" :
            
            self.derive_Z = []
            """PEUT ETRE COMMENCER A 1 PLUTOT QU'A 0 PUISQUE LA DERIVE DE L INPUT NE NOUS INTERRESSE PAS"""
            for activation in self.a:
                dZ = activation*(1 - activation) #Ici on part du principe que
                self.derive_Z.append(dZ)
        else :
            raise ValueError("La fonction d'activation rentrée n'est pas valide")

    def derive_Cout(self, activation_ouput_layer):
        #activation_ouput_layer = self.a[self.nb_couches -1]
        if self.loss_type == "least_squares" :
            self.derive_cout_activation = (activation_ouput_layer - self.y_real)
        else :
            raise ValueError("La fonction d'activation rentrée n'est pas valide")

    def initialisation_Matrice_Poids(self):
        self.w =[]
        self.b =[]
        
        for j in range(0, self.nb_couches- 1):
            
            nb_neurone_couche_suivante = self.liste_neurone_par_couche[j+1]
            nb_neurone_cette_couche = self.liste_neurone_par_couche[j]
            b0 = np.random.uniform(low=-1.0, high=1.0, size=(nb_neurone_couche_suivante,1)) #on commence à j+1 car les input neurones n'ont pas de biais
            w0 = np.ones((nb_neurone_couche_suivante, nb_neurone_cette_couche))
            self.b.append(b0)
            for i in range(0, nb_neurone_couche_suivante ) : 
                w0[i,:]= np.random.uniform(low=-1.0, high=1.0, size=nb_neurone_cette_couche)
            
            self.w.append(w0)

    def feedInput(self) :
    
        nb_neurone_premiere_couche = self.liste_neurone_par_couche[0]
        a0 = np.ones((nb_neurone_premiere_couche, self.nb_batch ))
        for i in range(self.nb_batch) :    #feed input neurone
            image, label = Get_image_And_Label(i)
            a0[: ,i] = image
        #print("shape label : ", np.shape(label))
        self.a.append(a0)


    def forward(self, input) :
        
        self.Feed_InputLayer(input)

        for i in range(self.nb_couches-1) :
            #print("i : ", i)
            #print("len a  : ", np.shape(self.a[i]), "len w :", np.shape(self.w[i]), "len b ", np.shape(self.b[i]))
            new_z = (np.matmul(self.w[i], self.a[i]) + self.b[i])
        
            self.z.append(new_z)
            activations = np.vectorize(self.activation)
        
            new_a = activations(new_z)
            self.a.append(new_a) #On met un 1 à la fin pour le biais
        
        activation_output_layer = self.a[self.nb_couches -1]
        return activation_output_layer

    def backward(self, activation_ouput_layer):
        self.derive_Activation()
        self.derive_Cout(activation_ouput_layer)
        self.Compute_delta()
        self.Change_Biais_and_Weights()

    def Transform_Label_To_Vector(self) :
        nb_output_neurones = self.liste_neurone_par_couche[self.nb_couches - 1]
        
        self.y_real = np.zeros((nb_output_neurones, self.nb_batch))

        for i in range(self.nb_batch):
            index_label = int(self.label_batch[i])
            self.y_real[index_label, i ] = 1
    
    def Compute_Loss_Function(self): 
        #print("shape y real : ", np.shape(self.y_real), "shape y_predicted : ", np.shape(self.a[self.nb_couches-1]))
        y_predicted = self.a[self.nb_couches-1]

        if self.loss_type == "least_squares" :
            diff_matrices = (self.y_real - y_predicted)**2
            self.cost = np.sum(diff_matrices, axis =0)/2
            #self.cost = np.mean(self.cost)
            return np.mean(self.cost)
        else :
            raise ValueError("La fonction de cout rentrée n'est pas valide")
        print("Cout de la fonction : ", np.mean(self.cost))

    def Compute_delta(self):
        self.deltas = [0]*self.nb_couches
        derive_activation_outputlayer = self.derive_Z[self.nb_couches-1]
        delta_output_layer = derive_activation_outputlayer * self.derive_cout_activation
        self.deltas[self.nb_couches - 1] = delta_output_layer

        
        """VERIFIER SI CETTE BOUCLE FAIT BIEN TOUTES LES ITERATIONS"""
        for i in range(self.nb_couches-2, 0, -1):#On met -1 car ca va s'arrêter à 0
            #print("shape w ",i, " : ", np.shape(self.w[i]))
            #print("shape delta ",i+1, " : ", np.shape(self.deltas[i+1]))
            #print("shape biais ",i, " : ", np.shape(self.b[i]))
            #print("shape dervé Z ",i, " : ", np.shape(self.derive_Z[i]))
            delta = np.matmul(self.w[i].T,self.deltas[i+1])*self.derive_Z[i]
            self.deltas[i] = delta
            #print("shape delta ",i, " : ", np.shape(delta))
            #print("shape activation ",i, " : ", np.shape(self.a[i]))

    def Change_Biais_and_Weights(self):
        for i in range(self.nb_couches-1) :
            gradient_bais = np.mean(self.deltas[i+1], axis =1, keepdims= True)
            #print("Shape b :  ", np.shape(self.b[i]))
            #print("Shape self.lnr*gradient_bais :  ", np.shape(self.lnr*gradient_bais))
            self.b[i] = self.b[i] - self.lnr*gradient_bais
            new_shape_a = np.reshape(self.a[i], (1, np.shape(self.a[i])[0], np.shape(self.a[i])[1]))
            new_shape_delta = np.reshape(self.deltas[i+1], (np.shape(self.deltas[i+1])[0],1, np.shape(self.deltas[i+1])[1]))
            gradient_weight = np.mean(new_shape_delta*new_shape_a, axis=2)
            self.w[i] = self.w[i] - self.lnr*gradient_weight
        

    
    def Train_network(self):
        
        #self.Test_Network(self.data_image_pool.T, self.train_nb_correct, self.label_pool) #pour avoir l'epoch 0
        #self.Test_Network(self.image_test.T, self.test_nb_correct, self.label_test)
       
        for j in range(self.nb_epoch) :
            print("Epoch : ", j+1)
            np.random.shuffle(self.index_image_and_label) #On change la composition des batchs
            self.costs = np.zeros((6000))
            self.iteration_batch = 0
            for i in range(6000) :#Avec 3000 et 10 inputs par batch il aura parcouru la moitié des données
                images = self.Index_Images_For_New_Batch()
                self.Transform_Label_To_Vector() #Peut être tout transformer en une fois
                activation_ouput_layer = self.forward(images)
                self.backward(activation_ouput_layer)
                self.costs[i] = self.Compute_Loss_Function()
                self.iteration_batch=self.iteration_batch+1
                if (j*6000 + (i+1))%(600*self.nb_epoch) == 0:
                    self.interface.DisplayProgressionBar(j*6000 + (i+1),600*self.nb_epoch)
                    self.Test_Network(self.data_image_pool.T, self.train_nb_correct, self.label_pool)
                    self.Test_Network(self.image_test.T, self.test_nb_correct, self.label_test)
            print("Cout de la fonction : ", np.mean(self.costs))
            #self.Test_Network(self.data_image_pool.T, self.train_nb_correct, self.label_pool)
            #self.Test_Network(self.image_test.T, self.test_nb_correct, self.label_test)
        self.ComputeIndEchec()

    def Test_Network(self, array_image, liste_score_par_epochs, labels) :
        activation_ouput_layer = self.forward(array_image)
        self.predic_test_set = np.argmax(activation_ouput_layer, axis =0)
        nb_correct = self.predic_test_set == labels
        #print("nb_corrct: ", nb_correct)
        #print("predict test set : ",len(self.predic_test_set))
        #print("label_test : ",len(self.label_test))
        pourcentage_correct = ((np.sum(nb_correct))/len(nb_correct))*100
        liste_score_par_epochs.append(pourcentage_correct)
        print("Le taux de prediction correcte est de : ", pourcentage_correct, "%")
    
    
    def ComputeIndEchec(self) :
        prediction = self.a[self.nb_couches-1]
        self.prediction = np.argmax(prediction, axis =0)
        self.ind_echec = np.where(self.prediction != self.label_test)
        
    def Retrieve_Error_After_Training(self) :
        """prediction = self.a[self.nb_couches-1]
        self.prediction = np.argmax(prediction, axis =0)
        self.ind_echec = np.where(self.prediction != self.label_test)"""
        #self.labelEnEchec = self.label_test[ind_echec]
        #self.predictionEnEchec = self.prediction[ind_echec]
        labelNumbers =[]
        occurences =[]
        for i in range(10):
            falsePredictionForiNumber = np.extract(self.label_test[self.ind_echec] == i, self.prediction[self.ind_echec])
            counter = Counter(falsePredictionForiNumber) #Renvoie un dictionnaire avec les occurences de chaque valeur
            number = list(counter.keys())
            occurence = list(counter.values())
            #print("number " ,number)
            #print("occurence ", occurence)
            labelNumbers.append(number)
            occurences.append(occurence)
        return labelNumbers, occurences
        
        for i in range(20) :
            print("Vrai chiffre : ", label_echec[i])
            print("Prediction : ", prevision_echec[i])
        #joblib.dump(label_echec, "label_echec.pkl")
     
    def ToString(self):
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