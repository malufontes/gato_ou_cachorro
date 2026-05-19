from ultralytics import YOLO

# Carrega um modelo pré-treinado para classificação.
# "yolo11n-cls.pt" é:
# - yolo11: versão do modelo
# - n: nano (pequeno e rápido)
# - cls: modelo para classificação
# - .pt: arquivo com pesos pré-treinados
model = YOLO("yolo11n-cls.pt")

# Inicia o treinamento
model.train(
    data="dataset",   # pasta contendo train/ e val/
    epochs=100,        # número de épocas
    imgsz=128,        # redimensiona todas as imagens para 128x128
    batch=32,         # 32 imagens por lote
    project="runs",   # pasta base para salvar resultados
    name="cat_or_dog",  # nome do experimento
)

# Avalia o modelo no conjunto de validação
metrics = model.val()   

print("Treinamento concluído!")
print("Melhor modelo salvo em:")
print("runs/cat_or_dog/weights/best.pt")
print("Acurácia Top-1:", metrics.top1)