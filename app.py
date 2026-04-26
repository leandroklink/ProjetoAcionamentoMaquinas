import cv2 #open CV - pacote de codigos 
import mediapipe as mp #ferramentas de reconhecimento de imagens
import os
import numpy as np
import time
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier

#É NECESSÁRIO UTILIZAR O PYTHON COM A VERSAO NO MÁXIMO 3.10.    

#utilize pip install mediapipe para rodar o código

class DetectorRosto():
    #classe responsavel por reconhecer rostos
    def __init__(self):
        self.mp_face = mp.solutions.face_detection
        self.face = self.mp_face.FaceDetection()
        self.desenho = mp.solutions.drawing_utils

    def encontrar_rosto(self, img):

        h, w, _ = img.shape
        rosto_img = None

        #converter para RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        #processamento
        resultado = self.face.process(img_rgb)

        #desenho
        if resultado.detections:
            for rosto in resultado.detections:
                bbox = rosto.location_data.relative_bounding_box

                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                largura = int(bbox.width * w)
                altura = int(bbox.height * h)

                #evitar o erro de corte fora da imagem
                x = max(0, x)
                y = max(0, y)
                largura = max(0, largura)
                altura = max(0, altura)    

                x = max(0, x)
                y = max(0, y)

                # caixa do rosto
                cv2.rectangle(img, (x, y), (x+largura, y+altura), (0, 255, 0), 2)

                rosto_img = img[y:y+altura, x:x+largura]
                break
        return img, rosto_img       


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

    #contar maos
    def contar_maos(self):
        if self.resultado.multi_hand_landmarks:
            return len(self.resultado.multi_hand_landmarks)
        return 0

#carregar DataSet + PCA
def carregar_dataset():
    dados = []
    labels = []
    path = "dataset"

    for pessoa in os.listdir(path):
        for img_name in os.listdir(os.path.join(path, pessoa)):
            img_path = os.path.join(path, pessoa, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (100, 100))

            vetor = img.flatten()

            dados.append(vetor)
            labels.append(pessoa)

    return np.array(dados), labels


dados, labels = carregar_dataset()

pca = PCA(n_components=5)
dados_pca = pca.fit_transform(dados)

modelo = KNeighborsClassifier(n_neighbors=3)
modelo.fit(dados_pca, labels)


def main():
    #capturar o video pela webcam


    # carregar lista de autorizados
    with open("autorizados.txt") as f:
        autorizados = [linha.strip() for linha in f]



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
        img, rosto = detectorRosto.encontrar_rosto(img)
        


        nome = "Desconhecido"

        if rosto is not None:
            try:
                rosto_gray = cv2.cvtColor(rosto, cv2.COLOR_BGR2GRAY)
                rosto_gray = cv2.resize(rosto_gray, (100, 100))

                vetor = rosto_gray.flatten().reshape(1, -1)
                vetor_pca = pca.transform(vetor)

                distancias, indices = modelo.kneighbors(vetor_pca)

                distancia = distancias[0][0]
                indice = indices[0][0]

                nome = labels[indice]

                # Limite = mais para aceitar melhor, menos para aceitar menos
                LIMIAR = 2680
                print("Distancia:", distancia)
                
                if distancia > LIMIAR:
                    nome = "Desconhecido"
            except:
                nome = "Erro"

        num_maos = detectorMaos.contar_maos()


        #REGRA DOS 3 Segundos
        if nome == "Desconhecido":
            status = "Nao reconhecido"
            tempo_inicio = None

        elif nome not in autorizados:
            status = "Nao autorizado"
            tempo_inicio = None

        elif num_maos < 2:
            status = "Mostre as duas maos"
            tempo_inicio = None

        else:
            if tempo_inicio is None:
                tempo_inicio = time.time()

            if time.time() - tempo_inicio >= 3:
                status = "Equipamento liberado"
            else:
                status = "Validando..."


        cv2.putText(img, f'Usuario: {nome}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        cv2.putText(img, f'Status: {status}', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)


        #mostrar captura 
        cv2.imshow('Captura de imagem', img)

        # tempo de atualização da captura (1ms)
        if cv2.waitKey(1) & 0xFF == 27: #tecla ESC para sair
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()