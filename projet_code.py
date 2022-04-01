# M17 : ANALYSE DES DONNÉES 
# PROF : Mr.CHEFCHAOUNI
# COMPTE RENDU DU : MINI-PROJET
# RÉALISÉ PAR : CHARAF YOUSRI
# RÉALISÉ LE : 14 JANVIER 2022
# RENDU AVANT : 30 JANVIER 2022


# Dans ce fichier , vous allez trouver le code qui repends au question du mini-projet ainsi que leur interpretation et explication .





'''    Importation des bibliothéque demandé    '''
from PIL import Image # pour pouvoir utiliser des images ;
import numpy as np  # pour utiliser les arrays prédefinies ;
import numpy.linalg as alg # pour pouvoir utiliser les fonctions de decomposition ;
from matplotlib import pyplot as plt # pour pouvoir afficher les résultats obtenus ; 
import matplotlib as mpl





'''*****************************************  PARTIE  I : application sur une image grise  *****************************************'''





'''PARTIE 01 : Lire et convertion de l'image , décomposition de la matrice  '''
im=Image.open("lena_gris.png") # pour lire l'image importé ;
T=np.array(im) # pour convertir l'image en matrice ;
m,n=T.shape  # on affect les dimension de la matrice au variable m et n ( m : nombre de lignes , n : nombre de colonnes ) ;
U, S, VT = alg.svd(T,full_matrices=False)  # on decompose la matrice T en fonction de  3 matrices ( U : m*m ; S : m*n ; VT : n*n ) ;
temp = S # on affect la valeur de S a une variable temprelle pour qu'on puisse l'utiliser aprés ;  
S = np.diag(S)





'''PARTIE 02 : création de la fonction compression '''
def compression(M,k): # la fontion prends comme variable une matrice et le nombre de valeur propres qui doivent rester apés compression ;
    U, S, VT = alg.svd(M,full_matrices=False) # on decompose la matrice en trois matrice ( U , S , VT ) ;
    S = np.diag(S) # la fonction diag retourn une matrice diagonale apartir du vecteur S ;
    M2= U[:,:k] @ S[:k,:k] @ VT[:k,:] # on veut garder juste les k premiéres valeurs singuliére , donc on prend juste les sub_matrixe utiles des 3 matrices  ( ells devienent : U : m*k , S : k*k , VT : k*n ) ;
    myimage=Image.fromarray(M2) # on converti la matrice M2 en image ;
    return myimage # on affiche l'image compréssé ;





'''PARTIE 03 : '''
#Plotting the compressed b&w image of Lena 
plt.figure()
k=30
fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(compression(T,k))
a.set_title('IMAGE COMPRÉSSÉ DE LEAN EN NOIRE ET BLANC POUR k = {}'.format(k))
plt.show()





'''PARTIE 04 : Affichage des images compressés selon les valuer de K '''
j=0
P=[] # on créer une liste vide
for j in range(10,160,10): # on boucle sur les valeurs (10 , 20 ,..., 140 , 150) ;
    P.append(compression(T,j)) # a chaque tour de boucle , on ajoute une image compréssé a notrs liste ;
fig = plt.figure(figsize=(20, 14)) # on cree la figure qui va nois afficher les 15 imges pour des differente valeurs de K ;
columns = 3 # le nombres d'images affichée par lignes ;
rows = 5 # le nombre de ligne d'affichage ;
for i in range(15):
  fig.add_subplot(rows, columns, i+1) # on ajoute un 'sub_plot' a la i_eme position
  plt.imshow(P[i]) #on affiche la i_eme image
  plt.axis('off') #pour ne pas afficher des axes(abs / ord )
  k=(i+1)*10
  compressionRatio = m*n / (k * (1 + m + n)) # la realtion du taux de compression ( uncompressed size / compressed size )
  plt.title("Picture N° {}, Taux de compression {}".format(i+1, round(compressionRatio, 3 ))) # pour ajouter un titre a chaque image (sub-plot)
  
  




'''PARTIE 05 :  afficher le graphe des valeurs singuliéres '''
fig, ax = plt.subplots()  # on veut afficher les valeur singuléres en fonction de k ;
x = [i for i in range(len(temp))] # l'axe des abscisse ;
y = temp # l'axe des ordonnées ;
plt.plot( x , y )
plt.xlabel("le nombre de valeurs singuliére (k) ")
plt.ylabel("les valeurs singuliéres") 
plt.title("les valeurs singulières en fnction de k") # pour afficher un titre a l'image 





'''PARTIE 06 : calcule de k qui donne .95 de la variance '''
for k in range(temp.shape[0]): # on parcour le vecteur qui contient les valeurs singuliéres
  if ( (sum(temp[:k])**2 / sum(temp)**2) >= 0.95): #  
    #on cherche le K qui verifie que : 
    # [(le carrées de la somme des k 1er valeurs singuliére) / ( le carrées de la sommes de toutes les valeurs singuléres)] > .95
    print("la valeur de k est : " , k) # on affiche le K verifiant la relation ;
    break;
plt.figure()
fig = plt.figure()
a = fig.add_subplot(1,1,1)
imgplot = plt.imshow(Image.fromarray(U[:,:k] @ S[:k,:k] @ VT[:k,:])) # on affiche l'image compréssées qui correspond au 1er k valeurs
a.set_title("l'image compressée qui correspond a 95% de lla variance")
#plt.axis('off')
plt.show()















'''*****************************************  PARTIE  II : application sur une image couleurs  *****************************************'''





image=Image.open("lena.png") # pour lire l'image importé ;
T_couleur=np.array(image) # pour convertir l'image en matrice ;
L,l,h =T_couleur.shape  # on affect les dimenssions de la matrice au L,l,h ( largeur , longueur , hauteur )
U_couleur , S_couleur , VT_couleur = alg.svd(T_couleur,full_matrices=False) # on decompose la matrice en trois autres matrice ;


'''    ploter l'image original de lena en RGB    '''
plt.figure()
fig = plt.figure()
a = fig.add_subplot(1,1,1)
imgplot = plt.imshow(Image.fromarray(T_couleur))
a.set_title('Lena couleur')
#plt.axis('off')
plt.show()





''' fonction de compression '''
def compression_couleur(image,k): 
  #la fonction prend comme argument une matrice 3d(qui represente une image couleur) et le nombre de valeurs singuliéres qu'on veut garder
  #la foncion retourne une matrice qui represente l'image ne couleur compréssée 
  compressed_image = np.zeros(image.shape) # on creer la matrice du retoure qui a les méme dimenssion 
  # dans cette fonction , on fait appel 3 fois de la fonction compression , qu'on a utilisé dans la partie du "noire et blanc"
  compressed_image[:,:,0]=compression(image[:, :, 0], k) # on applique la comprssion sur la couche rouge de l'image
  compressed_image[:,:,1]=compression(image[:, :, 1], k) # on applique la comprssion sur la couche verte de l'image
  compressed_image[:,:,2]=compression(image[:, :, 2], k) # on applique la comprssion sur la couche bleue de l'image
  # la matrice 'compressed_image' contient des valeurs negatives et des valeurs supérieur a 255 ,  ce qui affiche des taches verte
  # sur la photo , il faut les éliminer pour avoire une image plus claire; 
  compressed_image[compressed_image > 255] = 255 # on réduit toutes les valeurs qui dépasse 255 a la valeur 255;
  compressed_image[compressed_image < 0] = 0 # on ajuste toutes les valeurs negatives on l'affectent la valeurs 0;
  return compressed_image # on retourne la matrice





'''    ploter l'image compressée de lena en couleur avec la valeur de  k = 30    '''
plt.figure()
k=30
fig = plt.figure()
a = fig.add_subplot(1, 1, 1)
imgplot = plt.imshow(Image.fromarray(compression_couleur(T_couleur,k).astype('uint8')))
a.set_title('Image compréssée de Lena en couleur pour la valuer de  k = {}'.format(k))
#plt.axis('off')
plt.show()





'''    affichage des images comprésses en fonction des valeurs de k  ainsi que leur taux de compression   '''
# on utilise le méme algoirthme qu'on a utilisé dans l'image en noire et blanc 
j=0
l=[] # on créer une liste vide
for j in range(10,160,10): # on boucle sur les valeurs (10 , 20 ,..., 140 , 150) ;
    l.append(Image.fromarray(compression_couleur(T_couleur,j).astype('uint8'))) # a chaque tour de boucle , on ajoute une image compréssé a notrs liste ;
fig = plt.figure(figsize=(20, 14)) # on cree la figure qui va nois afficher les 15 imges pour des differente valeurs de K ;
columns = 3 # le nombres d'images affichée par lignes ;
rows = 5 # le nombre de ligne d'affichage ;
for i in range(15):
  fig.add_subplot(rows, columns, i+1) # on ajoute un 'sub_plot' a la i_eme position
  plt.imshow(l[i]) #on affiche la i_eme image
  plt.axis('off')
  k=(i+1)*10
  compressionRatio = m*n / (k * (1 + m + n )) # la realtion du taux de compression ( uncompressed size / compressed size )
  plt.title("Picture N° {}, Taux de compression {}".format(i+1, round(compressionRatio, 3 ))) # pour ajouter un titre a chaque image (sub-plot)










''' cette partie n'est pas demandé dans le rendu '''
# puisque la matrice est en couleur (RGB),on la decoupe en trois matrice qui represente les 3 niveau de couleur (rouge , verte , bleue) 
image_rouge = T_couleur[:, :, 0]
image_verte = T_couleur[:, :, 1]
image_blue  = T_couleur[:, :, 2]

# on decompose chaque chaque matrice (rouge , verte , bleu)
U_r, S_r, VT_r = np.linalg.svd(image_rouge)
U_v, S_v, VT_v = np.linalg.svd(image_verte)
U_b, S_b, VT_b = np.linalg.svd(image_blue)
   
# ploter l'image de lena au niveau du rouge
plt.figure()
fig = plt.figure()
a = fig.add_subplot(1,1,1)
imgplot = plt.imshow(Image.fromarray(image_rouge.astype('uint8')))
a.set_title('Lena au niveau du rouge')
#plt.axis('off')
plt.show()

# ploter l'image de lena au niveau du vert 
plt.figure()
fig = plt.figure()
a = fig.add_subplot(1,1,1)
imgplot = plt.imshow(Image.fromarray(image_verte.astype('uint8')))
a.set_title('Lena au niveau du vert')
#plt.axis('off')
plt.show()

# ploter l'image de lena au niveau du bleu
plt.figure()
fig = plt.figure()
a = fig.add_subplot(1,1,1)
imgplot = plt.imshow(Image.fromarray(image_blue.astype('uint8')))
a.set_title('Lena au niveau du blue')
#plt.axis('off')
plt.show()
