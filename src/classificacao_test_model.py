from pathlib import Path
from ultralytics import YOLO


# =========================
# CONFIGURAÇÕES
# =========================
MODEL_PATH = Path("runs/classify/runs/cat_or_dog/weights/best.pt")
TEST_DIR = Path("dataset/test")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_paths() -> None:
    """
    Verifica se o modelo e o dataset de teste existem.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado: {MODEL_PATH}\n"
            "Verifique se o treinamento foi concluído com sucesso."
        )

    if not TEST_DIR.exists():
        raise FileNotFoundError(
            f"Pasta de teste não encontrada: {TEST_DIR}"
        )

    class_dirs = [p for p in TEST_DIR.iterdir() if p.is_dir()]
    if not class_dirs:
        raise ValueError(
            "Nenhuma classe encontrada em dataset/test"
        )

    print("Estrutura de teste validada com sucesso!")
    print("Classes encontradas:",
          [p.name for p in class_dirs])


def is_image_file(path: Path) -> bool:
    """
    Retorna True se o arquivo for uma imagem suportada.
    """
    return path.suffix.lower() in VALID_EXTENSIONS


def main():
    print("=" * 60)
    print("AVALIAÇÃO DO MODELO NO CONJUNTO DE TESTE")
    print("=" * 60)

    # 1. Verifica se tudo existe
    validate_paths()

    # 2. Carrega o modelo
    print(f"\nCarregando modelo: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 3. Inicializa contadores
    total = 0
    correct = 0
    errors = []

    # 4. Percorre as classes (adult, kitten)
    for class_dir in sorted(TEST_DIR.iterdir()):
        if not class_dir.is_dir():
            continue

        true_label = class_dir.name
        print(f"\nTestando classe: {true_label}")

        # 5. Percorre as imagens da classe
        for image_path in sorted(class_dir.iterdir()):
            if not image_path.is_file():
                continue

            if not is_image_file(image_path):
                continue

            # Faz a predição
            results = model(image_path, verbose=False)

            # Classe prevista
            predicted_index = results[0].probs.top1
            predicted_label = results[0].names[predicted_index]

            # Confiança da previsão
            confidence = results[0].probs.top1conf.item()

            # Atualiza contadores
            total += 1

            if predicted_label == true_label:
                correct += 1
            else:
                errors.append({
                    "filename": image_path.name,
                    "expected": true_label,
                    "predicted": predicted_label,
                    "confidence": confidence,
                })

    # 6. Calcula métricas
    if total == 0:
        raise ValueError("Nenhuma imagem encontrada em dataset/test")

    accuracy = correct / total
    num_errors = total - correct

    # 7. Exibe resumo
    print("\n" + "=" * 60)
    print("RESULTADO FINAL")
    print("=" * 60)
    print(f"Total de imagens: {total}")
    print(f"Acertos:          {correct}")
    print(f"Erros:            {num_errors}")
    print(f"Acurácia:         {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # 8. Interpretação automática
    print("\nInterpretação:")
    if accuracy >= 0.95:
        print("Excelente desempenho.")
    elif accuracy >= 0.90:
        print("Muito bom desempenho.")
    elif accuracy >= 0.85:
        print("Bom desempenho.")
    elif accuracy >= 0.80:
        print("Desempenho aceitável.")
    else:
        print("Desempenho abaixo do ideal. Considere melhorar o dataset.")

    # 9. Mostra alguns erros
    if errors:
        print("\nExemplos de erros (até 10):")
        for error in errors[:10]:
            print(
                f"- {error['filename']}: "
                f"esperado={error['expected']}, "
                f"previsto={error['predicted']}, "
                f"confiança={error['confidence']:.2%}"
            )
    else:
        print("\nNenhum erro encontrado!")

    print("=" * 60)


if __name__ == "__main__":
    main()