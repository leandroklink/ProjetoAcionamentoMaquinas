import cv2 #open CV - pacote de codigos 
import mediapipe as mp #ferramentas de reconhecimento de imagens

class DetectorMaos():
    #classe responsavel pela detecçao de maos
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                rastreio_confianca=0.5, cor_pontos=(0,0,255), cor_conexoes=(255,255,255)):
        #funcao responsavel por inicializar a classe.
        #param modo: True a detecção é feita a todo momento, porém trava mais o software; False a detecçao não é feita a todo momento, podendo perder leituras, mas fica mais leve
        #param max_maos=2: maximo de mãos
        #param deteccao_condfianca: percentual da taxa de detecção da mao. Se for menor que esse limite a detecção não ocorre
        #param rastreio_confianca: percentual da taxa de rastreio da mao. Se for menor que esse limite a detecção não ocorre
        #param cor_pontos: cor dos pontos
        #param cor_conexoes: cor das conexões

        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_coinfianca =  rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        #inicializacao dos modulos de deteção
        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            self.modo,
            self.max_maos,
            self.deteccao_confianca,
            self.rastreio_coinfianca,
            self.cor_pontos,
            self.cor_conexoes
        )

        #Função para desenhar os pontos nas mãos
        self.desenho_mp = mp.solutions.drawing_utils

        #Configuracao do dsesenho dos pontos
        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(color=self.cor_pontos)

        #Configuracao do dsesenho das conexoes
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(color=self.cor_conexoes)

    def encontrar_maos(self, img, desenho=True):
        #Função responsavel por detectar as maos
        #param desenho: desenha os pontos e as conexoes, True para mostrar e false para nao
        #param img: imagem capturada


        #Converter imagem de BGR para RGB abaixo
        imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        self.resultado = self.maos.process(imagem_rgb)
        
        #verificar se alguma mao foi detectada
        if self.resultado.multi_hans_landmarks:
            for pontos in self.resultado.multi_handmarks:
                #desenhar os pontos nas mãos detectadas
                if desenho:
                    self.desenho_mp.draw_landmarks(
                        img, #imagem de captura
                        pontos, #pontos da mao
                        self.maos_mp.HAND_CONNECTIONS, #conexão entre os pontos
                        self.desenho_config_pontos, #cor dos pontos
                        self.desenho_config_conexoes #cor das conexoes

                    )
            return img

    def encontrar_pontos(): #funcao futura
        return


def main():
    #capturar o video pela webcam

    cap = cv2.VideoCapture(0) #identificar a webcan

    #instanciar a classe do detector
    detector = DetectorMaos(cor_pontos=(255, 0, 0), cor_conexoes=(255, 0, 0)) #definir a cor de pontos e conexoes


    #realizar captura
    while True:

        #obter imagem
        img = cap.read()

        #inverter a imagem
        img = cv2.flip(img, 1)


        img = detector.encontrar_maos(img)


        #mostrar captura 
        cv2.imshow('Captura de imagem', img)

        # tempo de atualização da captura
        cv2.waitKey(1)

if __name__ == '__main__':

    main()
                