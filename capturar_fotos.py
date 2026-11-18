#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Captura fotos da cabeca para treinar o reconhecimento do abafador.

Uso:  venv\\Scripts\\python capturar_fotos.py

Teclas (na janela):
    1 = gravar "abafador NA ORELHA"
    2 = gravar "abafador LEVANTADO" (preso no capacete, fora da orelha)
    3 = gravar "SEM abafador"
    ESPACO = pausar      Q = sair

Enquanto grava, mexa a cabeca devagar: frente, esquerda, direita, cima, baixo,
perto e longe da camera. Meta: 150 fotos ou mais por situacao.
"""
import sys
import time
from pathlib import Path

import cv2

import monitor_epi as M

CLASSES = {ord("1"): "na_orelha", ord("2"): "levantado", ord("3"): "sem_abafador"}
ROTULO = {"na_orelha": "1 NA ORELHA", "levantado": "2 LEVANTADO", "sem_abafador": "3 SEM ABAFADOR"}
PASTA = M.AQUI / "dataset_abafador"
INTERVALO = 0.25      # segundos entre fotos
META = 150


def contar():
    return {c: len(list((PASTA / c).glob("*.jpg"))) if (PASTA / c).exists() else 0
            for c in CLASSES.values()}


def main():
    cfg = M.carregar_config(M.AQUI / "config.yaml")
    cap = M.abrir_video(cfg["camera"])
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("largura", 1280))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("altura", 720))
    print("Carregando modelos...")
    pessoas = M.DetectorPessoas(cfg)
    rostos = M.Rostos(cfg["identificacao"])
    for c in CLASSES.values():
        (PASTA / c).mkdir(parents=True, exist_ok=True)
    qtd = contar()
    atual, ultimo = None, 0.0
    fonte = cv2.FONT_HERSHEY_SIMPLEX
    print(__doc__)
    while True:
        ok, quadro = cap.read()
        if not ok:
            time.sleep(0.1)
            continue
        tela = quadro.copy()
        dets = pessoas.rastrear(quadro)
        recorte = None
        if dets:
            # maior pessoa na imagem
            _, caixa = max(dets, key=lambda d: (d[1][2] - d[1][0]) * (d[1][3] - d[1][1]))
            face = None
            for f in rostos.detectar(quadro):
                cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
                if caixa[0] <= cx <= caixa[2] and caixa[1] <= cy <= caixa[3]:
                    if face is None or f[2] * f[3] > face[2] * face[3]:
                        face = f
            recorte, (x1, y1, x2, y2) = M.recorte_cabeca(quadro, caixa, face)
            cv2.rectangle(tela, (x1, y1), (x2, y2), M.AMARELO, 2)
        agora = time.monotonic()
        if atual and recorte is not None and recorte.size and agora - ultimo >= INTERVALO:
            ultimo = agora
            nome = f"{time.strftime('%Y%m%d_%H%M%S')}_{int((agora % 1) * 1000):03d}.jpg"
            cv2.imwrite(str(PASTA / atual / nome), recorte)
            qtd[atual] += 1

        cv2.rectangle(tela, (0, 0), (470, 118), (30, 30, 30), -1)
        y = 24
        for c in CLASSES.values():
            cor = M.VERDE if qtd[c] >= META else M.BRANCO
            marca = "  << GRAVANDO" if c == atual else ""
            cv2.putText(tela, f"{ROTULO[c]}: {qtd[c]}/{META}{marca}", (8, y), fonte, 0.6,
                        M.VERMELHO if c == atual else cor, 2 if c == atual else 1)
            y += 26
        cv2.putText(tela, "ESPACO = pausar   Q = sair", (8, y + 4), fonte, 0.5, M.CINZA, 1)
        if dets and recorte is None or not dets:
            cv2.putText(tela, "Ninguem na imagem", (8, tela.shape[0] - 14), fonte, 0.7, M.AMARELO, 2)
        cv2.imshow("Captura de fotos - abafador", tela)
        k = cv2.waitKey(1) & 0xFF
        if k in CLASSES:
            atual = CLASSES[k]
        elif k == ord(" "):
            atual = None
        elif k in (ord("q"), ord("Q"), 27):
            break
    cap.release()
    cv2.destroyAllWindows()
    print("\nFotos por situacao:", contar())
    print(f"Pasta: {PASTA}")
    if min(contar().values()) < 60:
        print("Atencao: alguma situacao tem menos de 60 fotos. Capture mais antes de treinar.")
    else:
        print("Agora rode:  venv\\Scripts\\python treinar_abafador.py")


if __name__ == "__main__":
    sys.exit(main())
