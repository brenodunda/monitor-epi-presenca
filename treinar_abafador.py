#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Treina o modelo que reconhece o abafador (na orelha / levantado / sem).

Uso:  venv\\Scripts\\python treinar_abafador.py

Le as fotos de dataset_abafador (criadas por capturar_fotos.py) e grava o
modelo em modelos\\abafador_cls.pt. O monitor passa a usa-lo sozinho.
Demora alguns minutos no notebook (sem placa de video).
"""
import shutil
from pathlib import Path

AQUI = Path(__file__).resolve().parent
ORIGEM = AQUI / "dataset_abafador"
TREINO = AQUI / "dataset_abafador_treino"
SAIDA = AQUI / "modelos" / "abafador_cls.pt"


def preparar():
    classes = sorted(d.name for d in ORIGEM.iterdir() if d.is_dir() and any(d.glob("*.jpg")))
    if len(classes) < 2:
        raise SystemExit("Preciso de fotos em pelo menos 2 situacoes. Rode capturar_fotos.py.")
    if TREINO.exists():
        shutil.rmtree(TREINO)
    for c in classes:
        fotos = sorted((ORIGEM / c).glob("*.jpg"))      # ordem de captura
        # validacao = ultimos 20% (fotos seguidas sao quase iguais; separar por
        # trecho evita uma nota de acerto enganosa)
        n_val = max(1, int(len(fotos) * 0.2))
        for parte, lista in (("train", fotos[:-n_val]), ("val", fotos[-n_val:])):
            destino = TREINO / parte / c
            destino.mkdir(parents=True, exist_ok=True)
            for f in lista:
                shutil.copy2(f, destino / f.name)
        print(f"  {c}: {len(fotos) - n_val} para treino, {n_val} para validacao")
    return classes


def main():
    from ultralytics import YOLO
    print("Preparando fotos...")
    preparar()
    modelo = YOLO("yolo11n-cls.pt")
    modelo.train(data=str(TREINO), epochs=25, imgsz=224, batch=16, workers=0,
                 device="cpu", patience=8, project=str(AQUI / "treinos"),
                 name="abafador", exist_ok=True, verbose=False)
    melhor = Path(modelo.trainer.save_dir) / "weights" / "best.pt"
    SAIDA.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(melhor, SAIDA)
    try:
        acerto = float(modelo.trainer.metrics.get("metrics/accuracy_top1", 0)) * 100
        print(f"\nAcerto na validacao: {acerto:.0f}%")
    except Exception:  # noqa: BLE001
        pass
    print(f"Modelo salvo em: {SAIDA}")
    print("Agora rode:  venv\\Scripts\\python monitor_epi.py --diagnostico")


if __name__ == "__main__":
    main()
