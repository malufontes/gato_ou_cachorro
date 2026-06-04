# Classificação de Gato ou Cachorro com YOLO

Projeto simples de classificação de imagens usando **Ultralytics YOLO** para identificar se uma imagem é de **gato** ou **cachorro**.

## Estrutura do projeto

```bash

GATO_OU_CACHORRO/
├── dataset/
│   └── classificacao/
│       ├── train/
│       ├── val/
│       └── test/
├── runs/
│   └── classify/
├── src/
│   ├── classificacao.ipynb
│   ├── classificacao.py
│   └── test_model.py
├── yolo11n-cls.pt
├── requirements.txt
└── .gitignore

```

## Dataset

O dataset está organizado no formato de classificação:

- `train/` → treinamento
- `val/` → validação
- `test/` → teste

Cada pasta contém subpastas com as classes:

- `gato`
- `cachorro`

## Modelo utilizado

O projeto usa o modelo pré-treinado:

```python
yolo11n-cls.pt
```

Foram treinados e testados modelos com diferentes configurações para testar a sua eficácia no jupyter notebook.

## Funcionalidades

- treinamento do modelo
- validação
- avaliação no conjunto de teste
- predição em imagens individuais

## Arquivos principais

- `src/classificacao.ipynb` → notebook com treino, avaliação e testes
- `src/classificacao.py` → script de treinamento
- `src/test_model.py` → script para avaliação do modelo

## Como executar

### 1. Instalar dependências

```bash
pip install -r requirements.txt
```

### 2. Treinar o modelo

Execute o notebook ou rode o script de treinamento.

### 3. Avaliar no conjunto de teste

Use o código de avaliação para medir a acurácia no `test/`.

### 4. Testar uma imagem

Use uma imagem isolada da pasta `raw/` para prever se é gato ou cachorro.

## Exemplo de saída

```text
cachorro: 18.80%
gato: 81.20%
```

Neste caso, a imagem foi classificada como **gato** com uma confiança de 81.20%.