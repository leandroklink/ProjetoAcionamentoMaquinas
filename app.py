import cv2 #open CV - pacote de codigos 
import mediapipe as mp #ferramentas de reconhecimento de imagens


#É NECESSÁRIO UTILIZAR O PYTHON COM A VERSAO NO MÁXIMO 3.10.    

#utilize pip install mediapipe para rodar o código

class DetectorMaos():
    #classe responsavel pela detecçao de maos
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                rastreio_confianca=0.5, cor_pontos=(0,0,255), cor_conexoes=(255,255,255)):
        #funcao responsavel por inicializar a classe.
        #param modo: True a detecção é feita a todo momento, porém trava mais o software; False a detecçao não é feita a todo momento, podendo perder leituras, mas fica mais leve
        #param max_maos=2: maximo de mãos
        #param deteccao_confianca: percentual da taxa de detecção da mao. Se for menor que esse limite a detecção não ocorre
        #param rastreio_confianca: percentual da taxa de rastreio da mao. Se for menor que esse limite a detecção não ocorre
        #param cor_pontos: cor dos pontos
        #param cor_conexoes: cor das conexões

        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        #inicializacao dos modulos de deteção
        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.max_maos,
            min_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )

        #Função para desenhar os pontos nas mãos
        self.desenho_mp = mp.solutions.drawing_utils

        #Configuracao do desenho dos pontos
        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(color=self.cor_pontos)

        #Configuracao do desenho das conexoes
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(color=self.cor_conexoes)

    def encontrar_maos(self, img, desenho=True):
        #Função responsavel por detectar as maos
        #param desenho: desenha os pontos e as conexoes, True para mostrar e false para nao
        #param img: imagem capturada

        #Converter imagem de BGR para RGB (necessario para o mediapipe)
        imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #processar a imagem
        self.resultado = self.maos.process(imagem_rgb)
        
        #verificar se alguma mao foi detectada
        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                #desenhar os pontos nas mãos detectadas
                if desenho:
                    self.desenho_mp.draw_landmarks(
                        img, #imagem de captura
                        pontos, #pontos da mao
                        self.maos_mp.HAND_CONNECTIONS, #conexão entre os pontos
                        self.desenho_config_pontos, #cor dos pontos
                        self.desenho_config_conexoes #cor das conexoes
                    )

        #retornar imagem (mesmo se nao detectar nada)
        return img

    def encontrar_pontos(self): #funcao futura
        return


def main():
    #capturar o video pela webcam

    cap = cv2.VideoCapture(0) #identificar a webcam

    #instanciar a classe do detector
    detector = DetectorMaos(cor_pontos=(255, 0, 0), cor_conexoes=(255, 0, 0)) #definir a cor de pontos e conexoes

    #realizar captura
    while True:

        #obter imagem da webcam
        sucesso, img = cap.read()

        #verificar se capturou corretamente
        if not sucesso:
            break

        #inverter a imagem (efeito espelho)
        img = cv2.flip(img, 1)

        #detectar mãos
        img = detector.encontrar_maos(img)

        #mostrar captura 
        cv2.imshow('Captura de imagem', img)

        # tempo de atualização da captura (1ms)
        if cv2.waitKey(1) & 0xFF == 27: #tecla ESC para sair
            break


if __name__ == '__main__':
    main()