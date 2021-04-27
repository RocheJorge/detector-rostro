from cv2 import cv2 #Importamos el archivo opencv

##################### Cargar Data Entrenada #############

data_rostro_entrenada = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#################### Captura Video #####################

##capturar un video en tiempo real

webCam = cv2.VideoCapture(0) #-> el 0 depende de la camara que se va a utilizar: 0 por defecto que trae la computadora, (1) otra camara, (2) (3), dependera de las camaras que tenemos instalas en la pc. Si no la posee marcara error

##Interactuar por tiempo indefinido con la camara

while True:
    ##leer los frames
    lectura_frame_exitosa, frame = webCam.read()
    
    ## convertir a escala de grises para sea mas facil la lectura para la maquina    
    
    bw_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) ####-> cvtColor permtie cambiar el color a la imagen (imagen, cv2.COLOR_BGR2GRAY). cv2.COLOR_BGR2GRAY permite cambiar el color a RGB gris

#################### Detectar Rostros #############

#####tomara las coordenadas donde detecte los rostros

    coordenadaRostro = data_rostro_entrenada.detectMultiScale(bw_img)
    #print(coordenadaRostro)

#################### Dibujar Rectangulos alrededor del rostro #############

    for (x, y, w, h) in coordenadaRostro:
       cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 2)
    
    ##mostrar captura de la camara
    cv2.imshow("Prueba de Mostrar Rostro", frame)
    key = cv2.waitKey(1) #si le paso un numero, esperara ese segundo para continuar grabando la pantalla, pero si no le paso nada esperara que presione una tecla para continuar con la captura de frames
    
    #presionamos la tecla q para salir del ciclo
    if key == 81 or key == 113:
        break

print("Codigo Finalizado")