Detector de Pessoa + Celular (OpenCV DNN + SSD MobileNet v3)

Este repositório contém um script em Python que usa OpenCV DNN com o modelo SSD MobileNet v3 (COCO) para detectar objetos em tempo real (webcam) ou a partir de um vídeo.
Ele destaca bounding boxes, exibe rótulos/confiança na tela e gera um alerta no console quando “person” e “cell phone” aparecem ao mesmo tempo.

✨ Funcionalidades

Detecção em tempo real via webcam (ou arquivo de vídeo).

Modelo leve e rápido (SSD MobileNet v3 treinado no COCO).

Desenho dos rótulos com cvzone (estética melhor que putText puro).

Regra de alerta: imprime alert! quando uma pessoa e um celular são detectados simultaneamente.

Supressão de não-máximos (NMS) para reduzir caixas sobrepostas.

🗂️ Estrutura sugerida do projeto
.
├── main.py
├── frozen_inference_graph.pb
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
├── coco.names
└── README.md


Renomeie seu arquivo principal para main.py (ou ajuste como preferir).

📦 Requisitos

Python 3.8+

Bibliotecas:

opencv-python

numpy

cvzone

Instalação rápida
# 1) Crie e ative um ambiente virtual (opcional, mas recomendado)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Instale as dependências
pip install opencv-python numpy cvzone



Arquivos do modelo

Coloque na raiz do repositório (ou ajuste os caminhos no código):

frozen_inference_graph.pb – pesos do SSD MobileNet v3 (COCO).

ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt – configuração do grafo.

coco.names – lista de classes do COCO (uma por linha).
Deve incluir, entre outras, as classes person e cell phone.

▶️ Como executar
Usando a webcam (padrão)

No código, videoPath = 0 usa a câmera padrão.

python main.py

Usando um arquivo de vídeo

Troque no código:

# videoPath = 'example1.mp4'
videoPath = 'SEU_ARQUIVO.mp4'


E rode:

python main.py

⚙️ Parâmetros principais no código
net.setInputSize(320, 320)             # Tamanho de entrada da rede (trade-off: velocidade x precisão)
net.setInputScale(1.0/127.5)           # Normalização
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)               # Converte BGR -> RGB

labels, confs, bboxs = net.detect(img, confThreshold=0.5)
# Aumente ou diminua confThreshold conforme necessário

cv2.dnn.NMSBoxes(bboxs, confs, score_threshold=0.5, nms_threshold=0.3)
# Ajuste nms_threshold se houver muitas caixas duplicadas

🔔 Lógica do alerta

No loop principal, o script cria dois conjuntos (person, cell) e adiciona as labels detectadas.
Se “person” e “cell phone” estiverem presentes ao mesmo tempo, imprime:

alert!


Você pode trocar essa ação por:

tocar um som,

enviar um webhook,

registrar em arquivo,

exibir um overlay, etc.
