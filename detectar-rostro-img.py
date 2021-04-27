from cv2 import cv2 #Importamos el archivo opencv

##################### Cargar Data Entrenada #############

data_rostro_entrenada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#################### Captura Imagen #############

###Escoger una imagen para detectar rostros

img = cv2.imread("img01.jpg") #-> detectara la imagen de rostros y la guardara en la variable img

####pasar a escala de negros la imagen para que la computadora la pueda leer mejor

bw_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) ####-> cvtColor permitie cambiar el color a la imagen (imagen, cv2.COLOR_BGR2GRAY). cv2.COLOR_BGR2GRAY permite cambiar el color a RGB gris

#################### Cambiar tamaño de la ventana si es muy grande #############

newImg = cv2.resize(img, (700, 700)) #-> En 700, 700 se pueden acomodar los pixeles de altura-anchura

#################### Detectar Rostros #############

#####tomara las coordenadas donde detecte los rostros

coordenadaRostro = data_rostro_entrenada.detectMultiScale(newImg)
#print(coordenadaRostro)

#################### Dibujar Rectangulos alrededor del rostro #############

for (x, y, w, h) in coordenadaRostro:
   cv2.rectangle(newImg, (x,y), (x+w, y+h), (0, 255, 0), 2)

#################### mostrar imagen #############

cv2.imshow("Detector de Rostros", newImg) #####-> imshow muesta la ventana donde aparecera la imagen de los rostros

cv2.waitKey() #####-> waitkey() esperara que yo presione una tecla para salir del programa

print("Codigo Finalizado")