import cv2 #open CV - pacote de codigos 
import mediapipe as mp #ferramentas de reconhecimento de imagens


#É NECESSÁRIO UTILIZAR O PYTHON COM A VERSAO NO MÁXIMO 3.10.    

#utilize pip install mediapipe cv2 para rodar o código

class DetectorRosto():
    #classe responsavel por reconhecer rostos
    def __init__(self, deteccao_confianca=0.5):
        
        self.deteccao_confianca = deteccao_confianca

        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection()
        self.desenho = mp.solutions.drawing_utils

    def encontrar_rosto(
            self,
            img, # Imagem capturada.
            desenho=True, # Desenhar a(s) caixa(s) de detecção do(s) rosto(s).
            cor_caixa=(255, 0, 255), # Cor da caixa.
            cor_moldura=(255, 0, 255), #Cor da moldura.
            comprimento=30, #Comprimento da linha da moldura.
            espessura_moldura=5, # Espessura da linha da moldura.
            espessura_retangulo=1, #Espessura do retângulo da(s) caixa(s).
            desenho_caixa=True, # Desenhar a caixa de detecção.
            desenho_moldura=True # Desenhar a moldura na caixa de detecção.
        ):

        #converter para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #processamento
        resultado = self.face.process(img_rgb)
        

        #lista com caixas de deteccao de rostos
        caixas = []


        #desenho
        if resultado.detections:
            for id, rosto in enumerate(resultado.detections):

                #informacoes da caixa de reconhecimento

                info_caixa = rosto.location_data.relative_bounding_box
                self.desenho.draw_detection(img, rosto)

                altura, largura, _ = img.shape

                #caixas de deteccao
                caixa = (int(info_caixa.xmin * largura), int(info_caixa.ymin * altura),
                         int(info_caixa.width * largura), int(info_caixa.height * largura))
                
                caixas.append([id, caixa, rosto .score])

                #desenhar caixa em volta do rosto
                if rosto:
                    img = self.desenho_moldura(
                        img, caixa, cor_caixa, cor_moldura, comprimento,
                        espessura_moldura, espessura_retangulo, desenho_caixa, desenho_moldura)
                                        # --- Colocar na caixa de detecção a porcentagem de certeza da detecção --- #
                    
                    cv2.putText(
                        img,  # imagem capturada
                        f'{int(rosto.score[0] * 100)}%',  # texto
                        (caixa[0], caixa[1] - 20),  # posição do texto
                        cv2.FONT_HERSHEY_PLAIN,  # fonte
                        2,  # tamanho da fonte
                        (255, 0, 255),  # cor da fonte
                        2  # espessura
                    )
        return img,caixas       
    

    def desenho_moldura(
            self, img, caixa, cor_caixa, cor_moldura, comprimento, espessura_moldura, 
            espessura_retangulo, desenho_caixa, desenho_moldura):
        # --- Dimensões da caixa --- #
        x, y, largura, altura = caixa

        # --- Pontos iniciais da moldura --- #
        x_1, y_1 = x + largura, y + largura

        if desenho_caixa:
            # --- Desenhar a caixa em volta do rosto detectado --- #
            cv2.rectangle(
                img,  # imagem capturada
                caixa,  # pontos da caixa
                cor_caixa,  # cor do retângulo
                espessura_retangulo  # espessura da linha do retângulo
            )

        if desenho_moldura:
            # --- Canto superior esquerdo --- #
            cv2.line(
                img,   # imagem capturada
                (x, y),  # ponto inicial
                (x + comprimento, y),  # ponto final
                cor_moldura,  # cor da moldura
                espessura_moldura  # espessura da moldura
            )
            cv2.line(
                img, 
                (x, y),
                (x, y + comprimento),
                cor_moldura,
                espessura_moldura
            )

            # --- Canto superior direito --- #
            cv2.line(
                img,
                (x_1, y),
                (x_1 - comprimento, y),
                cor_moldura,
                espessura_moldura
            )
            cv2.line(
                img,
                (x_1, y),
                (x_1, y + comprimento),
                cor_moldura,
                espessura_moldura
            )

            # --- Canto inferior esquerdo --- #
            cv2.line(
                img, 
                (x, y_1),
                (x + comprimento, y_1),
                cor_moldura,
                espessura_moldura
            )
            cv2.line(
                img,
                (x, y_1),
                (x, y_1 - comprimento),
                cor_moldura,
                espessura_moldura
            )

            # --- Canto inferior direito --- #
            cv2.line(
                img,
                (x_1, y_1),
                (x_1 - comprimento, y_1),
                cor_moldura,
                espessura_moldura
            )
            cv2.line(
                img,
                (x_1, y_1),
                (x_1, y_1 - comprimento),
                cor_moldura,
                espessura_moldura
            )

        return img


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
    detectorMaos = DetectorMaos(cor_pontos=(255, 0, 0), cor_conexoes=(255, 0, 0)) #definir a cor de pontos e conexoes
    detectorRosto = DetectorRosto()
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
        img = detectorMaos.encontrar_maos(img)
        #detectar rosto
        img, caixas = detectorRosto.encontrar_rosto(img)
        

        #mostrar captura 
        cv2.imshow('Captura de imagem', img)

        # tempo de atualização da captura (1ms)
        if cv2.waitKey(1) & 0xFF == 27: #tecla ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()