import cv2
import mediapipe as mp

class DetectorMaos():
    def __init__(self, modo=False, max_maos=2, deteccao_confianca=0.5,
                 rastreio_confianca=0.5, cor_pontos=(0,0,255), cor_conexoes=(255,255,255)):

        self.modo = modo
        self.max_maos = max_maos
        self.deteccao_confianca = deteccao_confianca
        self.rastreio_confianca = rastreio_confianca
        self.cor_pontos = cor_pontos
        self.cor_conexoes = cor_conexoes

        self.maos_mp = mp.solutions.hands
        self.maos = self.maos_mp.Hands(
            static_image_mode=self.modo,
            max_num_hands=self.max_maos,
            min_detection_confidence=self.deteccao_confianca,
            min_tracking_confidence=self.rastreio_confianca
        )

        self.desenho_mp = mp.solutions.drawing_utils

        self.desenho_config_pontos = self.desenho_mp.DrawingSpec(color=self.cor_pontos)
        self.desenho_config_conexoes = self.desenho_mp.DrawingSpec(color=self.cor_conexoes)

    def encontrar_maos(self, img, desenho=True):
        imagem_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.resultado = self.maos.process(imagem_rgb)

        if self.resultado.multi_hand_landmarks:
            for pontos in self.resultado.multi_hand_landmarks:
                if desenho:
                    self.desenho_mp.draw_landmarks(
                        img,
                        pontos,
                        self.maos_mp.HAND_CONNECTIONS,
                        self.desenho_config_pontos,
                        self.desenho_config_conexoes
                    )

        return img


def main():
    cap = cv2.VideoCapture(0)

    detector = DetectorMaos(cor_pontos=(255, 0, 0), cor_conexoes=(255, 0, 0))

    while True:
        sucesso, img = cap.read()

        if not sucesso:
            break

        img = cv2.flip(img, 1)

        img = detector.encontrar_maos(img)

        cv2.imshow('Captura de imagem', img)

        if cv2.waitKey(1) & 0xFF == 27:
            break


if __name__ == '__main__':
    main()