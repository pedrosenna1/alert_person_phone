# Detector de Pessoa + Celular (OpenCV DNN + SSD MobileNet v3)

Este repositório contém um script em Python que usa **OpenCV DNN** com o modelo **SSD MobileNet v3 (COCO)** para detectar objetos em tempo real (webcam) ou a partir de um vídeo.  
Ele destaca bounding boxes, exibe rótulos/confiança na tela e **gera um alerta no console quando “person” e “cell phone” aparecem ao mesmo tempo**.

---

## ✨ Funcionalidades

- Detecção em tempo real via webcam (ou arquivo de vídeo).
- Modelo leve e rápido (SSD MobileNet v3 treinado no COCO).
- Desenho dos rótulos com **cvzone**.
- **Regra de alerta**: imprime `alert!` quando uma pessoa e um celular são detectados simultaneamente.
- Supressão de não-máximos (**NMS**) para reduzir caixas sobrepostas.

---

## 🗂️ Estrutura sugerida do projeto

```
.
├── main.py
├── frozen_inference_graph.pb
├── ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
├── coco.names
└── README.md
```

---

## 📦 Requisitos

- Python 3.8+  
- Bibliotecas:
  - `opencv-python` (ou `opencv-contrib-python`)
  - `numpy`
  - `cvzone`

### Instalação rápida

```bash
# 1) Crie e ative um ambiente virtual (opcional)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2) Instale as dependências
pip install opencv-python numpy cvzone
```

### Arquivos do modelo

Coloque na raiz do repositório (ou ajuste os caminhos no código):

- `frozen_inference_graph.pb` – pesos do SSD MobileNet v3 (COCO).  
- `ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt` – configuração do grafo.  
- `coco.names` – lista de classes do COCO (uma por linha).

---

## ▶️ Como executar

### Usando a webcam (padrão)

```bash
python main.py
```

### Usando um arquivo de vídeo

Edite no código:

```python
videoPath = 'SEU_VIDEO.mp4'
```

E rode:

```bash
python main.py
```

---

## ⚙️ Parâmetros principais no código

```python
net.setInputSize(320, 320)             # Tamanho da entrada
net.setInputScale(1.0/127.5)           # Normalização
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)               # BGR -> RGB

labels, confs, bboxs = net.detect(img, confThreshold=0.5)
cv2.dnn.NMSBoxes(bboxs, confs, score_threshold=0.5, nms_threshold=0.3)
```

---

## 🔔 Lógica do alerta

O script cria dois conjuntos (`person`, `cell`).  
Se **“person”** e **“cell phone”** estiverem presentes ao mesmo tempo, imprime no console:

```
alert!
```

Você pode trocar essa ação por envio de notificação, log em arquivo, webhook, etc.

---

## 🧩 Código completo

```python
import cv2
import numpy as np
import cvzone

# videoPath = 'example1.mp4'
videoPath = 0
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
modelPath = 'frozen_inference_graph.pb'
classesPath='coco.names'

net = cv2.dnn_DetectionModel(modelPath,configPath)

net.setInputSize(320,320)
net.setInputScale(1.0/127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)

with open(classesPath,'r') as f:
    classesList = f.read().splitlines()

video = cv2.VideoCapture(videoPath)

while True:
    check,img = video.read()
    img = cv2.resize(img,(1270,720))

    labels, confs, bboxs = net.detect(img,confThreshold=0.5)

    bboxs = list(bboxs)
    confs = list(np.array(confs).reshape(1,-1)[0])
    confs = list(map(float,confs))

    bboxsIdx = cv2.dnn.NMSBoxes(bboxs,confs,score_threshold=0.5, nms_threshold=0.3)

    if len(bboxsIdx) !=0:
        person = set()
        cell = set()
        for x in range(0,len(bboxsIdx)):
            bbox = bboxs[np.squeeze(bboxsIdx[x])]
            conf = confs[np.squeeze(bboxsIdx[x])]
            labelsID = np.squeeze(labels[np.squeeze(bboxsIdx[x])])-1
            label = classesList[labelsID]

            if label == "person":
                x,y,w,h = bbox
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                cvzone.putTextRect(img,f'{label} {round(conf,2)}',(x,y-10),colorR=(255,0,0),scale=1,thickness=2)
                person.add(label)

            elif label == "cell phone":
                x,y,w,h = bbox
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),3)
                cvzone.putTextRect(img,f'{label} {round(conf,2)}',(x,y-10),colorR=(255,0,0),scale=1,thickness=2)
                cell.add(label)

        if "person" in person and "cell phone" in cell:
            print('alert!')

    cv2.imshow('Imagem',img)

    if cv2.waitKey(1)==27:
        break
```

---

## 🧪 Dicas

- **Sem vídeo**: verifique se `videoPath = 0` e a webcam está disponível.  
- **Erro de modelo**: confira caminhos de `.pb` e `.pbtxt`.  
- **Detecções fracas**: aumente `confThreshold`.  
- **Performance**: reduza resolução no `cv2.resize`.

---







