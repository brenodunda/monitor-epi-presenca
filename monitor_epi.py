#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Monitor de EPI e Presenca
=========================
- Detecta e rastreia cada pessoa na camera.
- Da um ID fixo por pessoa (P1, P2...) reconhecendo o rosto.
- Confere capacete, oculos e protetor auricular.
- Mede o tempo de presenca e de ausencia de cada pessoa.
- Gera alertas (tela, som, CSV, foto, Telegram, WhatsApp e webhook).

Uso:
    python monitor_epi.py                      -> abre a camera do config.yaml
    python monitor_epi.py --diagnostico        -> mostra o que o detector de EPI ve
    python monitor_epi.py --video teste.mp4    -> roda em um video gravado
    python monitor_epi.py --listar-cameras     -> descobre o numero de cada camera
    python monitor_epi.py --testar-telegram    -> configura/testa o alerta no Telegram
    python monitor_epi.py --listar             -> lista as pessoas cadastradas
    python monitor_epi.py --renomear P1 "Maria"
    python monitor_epi.py --apagar P3

Teclas na janela:  Q ou ESC = sair
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
import yaml

AQUI = Path(__file__).resolve().parent
PASTA_MODELOS = AQUI / "modelos"

URL_YUNET = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_detection_yunet/face_detection_yunet_2023mar.onnx")
URL_SFACE = ("https://github.com/opencv/opencv_zoo/raw/main/models/"
             "face_recognition_sface/face_recognition_sface_2021dec.onnx")

VERDE, VERMELHO, AMARELO, CINZA, BRANCO = (
    (80, 200, 80), (60, 60, 230), (0, 200, 255), (160, 160, 160), (255, 255, 255))


# ----------------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------------
def fmt_duracao(seg: float) -> str:
    """125 -> '2min 05s' ; 3725 -> '1h 02min 05s'"""
    seg = int(max(0, round(seg)))
    h, resto = divmod(seg, 3600)
    m, s = divmod(resto, 60)
    if h:
        return f"{h}h {m:02d}min {s:02d}s"
    if m:
        return f"{m}min {s:02d}s"
    return f"{s}s"


def baixar(url: str, destino: Path) -> Path:
    if destino.exists() and destino.stat().st_size > 10_000:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    print(f"Baixando {destino.name} ...")
    try:
        urllib.request.urlretrieve(url, destino)
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            f"\nNao consegui baixar {destino.name} ({e}).\n"
            f"Baixe manualmente em:\n  {url}\ne salve em:\n  {destino}\n")
    return destino


def iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0


def destinos_whatsapp(cfg_alertas: dict) -> list[tuple[str, str]]:
    """Le a lista alertas.whatsapp do config e devolve [(telefone, apikey)]."""
    saida = []
    for d in cfg_alertas.get("whatsapp") or []:
        tel = str((d or {}).get("telefone") or "").strip().replace(" ", "")
        chave = str((d or {}).get("apikey") or "").strip()
        if tel and chave:
            saida.append((tel, chave))
    return saida


def enviar_whatsapp(telefone: str, apikey: str, texto: str) -> bool:
    """Envia texto pelo CallMeBot (servico gratuito de terceiros, uso pessoal)."""
    url = "https://api.callmebot.com/whatsapp.php?" + urllib.parse.urlencode(
        {"phone": telefone, "text": texto, "apikey": apikey})
    try:
        resposta = urllib.request.urlopen(url, timeout=25).read().decode("utf-8", "ignore")
    except Exception as e:  # noqa: BLE001
        print(f"  (WhatsApp para {telefone[:6]}...: falha no envio: {e})", flush=True)
        return False
    if "error" in resposta.lower() or "invalid" in resposta.lower():
        print(f"  (WhatsApp para {telefone[:6]}...: recusado: {resposta[:200]})", flush=True)
        return False
    return True


def telegram_chamar(token: str, metodo: str, campos: dict, foto: Path | None = None) -> dict:
    """Chama a API oficial de bots do Telegram e devolve a resposta (dict)."""
    url = f"https://api.telegram.org/bot{token}/{metodo}"
    if foto is None:
        req = urllib.request.Request(url, data=urllib.parse.urlencode(campos).encode("utf-8"))
    else:
        limite = "----monitorepi" + str(int(time.time() * 1000))
        corpo = b""
        for k, v in campos.items():
            corpo += (f"--{limite}\r\nContent-Disposition: form-data; name=\"{k}\"\r\n\r\n"
                      f"{v}\r\n").encode("utf-8")
        corpo += (f"--{limite}\r\nContent-Disposition: form-data; name=\"photo\"; "
                  f"filename=\"alerta.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode("utf-8")
        corpo += foto.read_bytes() + f"\r\n--{limite}--\r\n".encode("utf-8")
        req = urllib.request.Request(
            url, data=corpo,
            headers={"Content-Type": f"multipart/form-data; boundary={limite}"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=30).read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            return json.loads(e.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {"ok": False, "description": f"HTTP {e.code}"}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "description": str(e)}


def config_telegram(cfg_alertas: dict) -> tuple[str, str]:
    t = cfg_alertas.get("telegram") or {}
    return str(t.get("token") or "").strip(), str(t.get("chat_id") or "").strip()


def enviar_telegram(cfg_alertas: dict, texto: str, foto: Path | None = None) -> bool:
    token, chat = config_telegram(cfg_alertas)
    if not token or not chat:
        return False
    if foto is not None and foto.exists():
        r = telegram_chamar(token, "sendPhoto", {"chat_id": chat, "caption": texto}, foto)
    else:
        r = telegram_chamar(token, "sendMessage", {"chat_id": chat, "text": texto})
    if not r.get("ok"):
        print(f"  (Telegram: falha no envio: {r.get('description')})", flush=True)
    return bool(r.get("ok"))


class Relogio:
    """Webcam: relogio real. Arquivo de video: tempo do proprio video."""

    def __init__(self, fps_arquivo: float | None):
        self.fps = fps_arquivo
        self.inicio = datetime.now()
        self.t0 = time.monotonic()
        self.quadro = 0

    def avancar(self):
        self.quadro += 1

    def t(self) -> float:
        if self.fps:
            return self.quadro / self.fps
        return time.monotonic() - self.t0

    def hora(self, t: float | None = None) -> datetime:
        return self.inicio + timedelta(seconds=self.t() if t is None else t)


# ----------------------------------------------------------------------------
# Cadastro de pessoas (galeria de rostos)
# ----------------------------------------------------------------------------
class Galeria:
    """Guarda so os vetores numericos do rosto (nao guarda fotos)."""
    MAX_AMOSTRAS = 12

    def __init__(self, arquivo: Path):
        self.arquivo = arquivo
        self.pessoas: dict[str, dict] = {}
        if arquivo.exists():
            bruto = json.loads(arquivo.read_text(encoding="utf-8"))
            for pid, p in bruto.items():
                p["vetores"] = [np.array(v, dtype=np.float32) for v in p["vetores"]]
                self.pessoas[pid] = p

    def salvar(self):
        self.arquivo.parent.mkdir(parents=True, exist_ok=True)
        saida = {pid: {"nome": p["nome"], "criado": p["criado"],
                       "vetores": [v.tolist() for v in p["vetores"]]}
                 for pid, p in self.pessoas.items()}
        self.arquivo.write_text(json.dumps(saida), encoding="utf-8")

    def nome(self, pid: str) -> str:
        p = self.pessoas.get(pid)
        return (p["nome"] or pid) if p else pid

    def comparar(self, vetor: np.ndarray) -> tuple[str | None, float]:
        """Devolve (id mais parecido, similaridade de cosseno)."""
        melhor, nota = None, -1.0
        for pid, p in self.pessoas.items():
            for v in p["vetores"]:
                s = float(np.dot(vetor, v))
                if s > nota:
                    melhor, nota = pid, s
        return melhor, nota

    def nova(self, vetores: list[np.ndarray], quando: datetime) -> str:
        n = 1
        while f"P{n}" in self.pessoas:
            n += 1
        pid = f"P{n}"
        self.pessoas[pid] = {"nome": "", "criado": quando.isoformat(timespec="seconds"),
                             "vetores": vetores[: self.MAX_AMOSTRAS]}
        self.salvar()
        return pid

    def reforcar(self, pid: str, vetor: np.ndarray, nota: float):
        """Guarda novas aparencias do mesmo rosto (com capacete, oculos...)."""
        p = self.pessoas[pid]
        if nota < 0.75 and len(p["vetores"]) < self.MAX_AMOSTRAS:
            p["vetores"].append(vetor)
            self.salvar()


class Rostos:
    """Detector (YuNet) + reconhecedor (SFace) do proprio OpenCV."""

    def __init__(self, cfg: dict):
        det = baixar(URL_YUNET, PASTA_MODELOS / "face_detection_yunet_2023mar.onnx")
        rec = baixar(URL_SFACE, PASTA_MODELOS / "face_recognition_sface_2021dec.onnx")
        criar_det = getattr(cv2.FaceDetectorYN, "create", None) or cv2.FaceDetectorYN_create
        criar_rec = getattr(cv2.FaceRecognizerSF, "create", None) or cv2.FaceRecognizerSF_create
        self.det = criar_det(str(det), "", (320, 320), 0.7, 0.3, 50)
        self.rec = criar_rec(str(rec), "")
        self.min_px = cfg["tamanho_minimo_rosto"]

    def detectar(self, quadro) -> list[np.ndarray]:
        h, w = quadro.shape[:2]
        self.det.setInputSize((w, h))
        _, faces = self.det.detect(quadro)
        return [] if faces is None else list(faces)

    def vetor(self, quadro, face) -> np.ndarray:
        alinhado = self.rec.alignCrop(quadro, face)
        v = self.rec.feature(alinhado).flatten().astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-9)


# ----------------------------------------------------------------------------
# Detectores (pessoas e EPI)
# ----------------------------------------------------------------------------
class DetectorPessoas:
    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        self.modelo = YOLO(cfg["modelo_pessoa"])

    def rastrear(self, quadro) -> list[tuple[int, tuple[int, int, int, int]]]:
        """Devolve [(id_rastreio, (x1, y1, x2, y2)), ...]"""
        r = self.modelo.track(quadro, persist=True, classes=[0], conf=0.4,
                              tracker="bytetrack.yaml", verbose=False)[0]
        if r.boxes is None or r.boxes.id is None:
            return []
        ids = r.boxes.id.int().tolist()
        caixas = r.boxes.xyxy.int().tolist()
        return [(i, tuple(c)) for i, c in zip(ids, caixas)]


class DetectorEPI:
    """Procura os EPIs no recorte da cabeca de cada pessoa."""

    def __init__(self, cfg: dict):
        from ultralytics import YOLO
        self.itens: dict = cfg["epis"]
        self.imgsz = cfg.get("tamanho_imagem_epi", 480)
        caminho = str(cfg["modelo_epi"])
        nome = Path(caminho).name.lower()
        descricoes = [d for item in self.itens.values() for d in item["descricoes"]]
        if "yoloe" in nome:
            from ultralytics import YOLOE
            self.modelo = YOLOE(caminho)
            self.modelo.set_classes(descricoes, self.modelo.get_text_pe(descricoes))
        elif "world" in nome:
            self.modelo = YOLO(caminho)
            self.modelo.set_classes(descricoes)
        else:  # modelo proprio treinado: usa as classes dele
            self.modelo = YOLO(caminho)
        self.conf_min = min(i["confianca"] for i in self.itens.values())
        self.diagnostico = False      # --diagnostico: mostra tudo que o detector ve
        # Classificadores treinados (opcional): substituem o detector generico no item
        self.classificadores: dict = {}
        for nome_item, item in self.itens.items():
            cam = item.get("classificador")
            if not cam:
                continue
            arq = Path(cam) if Path(cam).is_absolute() else AQUI / cam
            if arq.exists():
                self.classificadores[nome_item] = (
                    YOLO(str(arq)),
                    [str(c).lower() for c in item.get("classes_ok", [])],
                    float(item.get("confianca_classificador", 0.6)))
                print(f"Usando modelo treinado para '{nome_item}': {arq.name}")
        self.imagem_diag = None

    def conferir(self, recortes: list[np.ndarray]) -> list[dict[str, float]]:
        """Para cada recorte devolve {nome_do_epi: maior confianca encontrada}."""
        if not recortes:
            return []
        conf = 0.03 if self.diagnostico else self.conf_min
        resultados = self.modelo.predict(recortes, imgsz=self.imgsz,
                                         conf=conf, verbose=False)
        if self.diagnostico:
            try:
                self.imagem_diag = resultados[0].plot(masks=False)
            except TypeError:
                self.imagem_diag = resultados[0].plot()
        saida = []
        for r in resultados:
            achados: dict[str, float] = {}
            if r.boxes is not None:
                for c, p in zip(r.boxes.cls.int().tolist(), r.boxes.conf.tolist()):
                    classe = str(r.names[c]).lower()
                    for nome_item, item in self.itens.items():
                        if classe in [d.lower() for d in item["descricoes"]] \
                                and p >= item["confianca"]:
                            achados[nome_item] = max(p, achados.get(nome_item, 0.0))
            saida.append(achados)
        for nome_item, (modelo, classes_ok, conf_min) in self.classificadores.items():
            rs = modelo.predict(recortes, imgsz=224, verbose=False)
            for achados, r in zip(saida, rs):
                classe = str(r.names[int(r.probs.top1)]).lower()
                p = float(r.probs.top1conf)
                achados.pop(nome_item, None)
                if classe in classes_ok and p >= conf_min:
                    achados[nome_item] = p
                if self.diagnostico:
                    print(f"  [{nome_item}] {classe} ({p:.2f})", flush=True)
        return saida


def recorte_cabeca(quadro, caixa, face) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Recorta a regiao da cabeca (com folga para capacete e abafador)."""
    H, W = quadro.shape[:2]
    x1, y1, x2, y2 = caixa
    if face is not None:
        fx, fy, fw, fh = face[:4]
        cx = fx + fw / 2
        lado = max(fw, fh)
        rx1, rx2 = cx - 1.4 * lado, cx + 1.4 * lado
        ry1, ry2 = fy - 1.1 * lado, fy + fh + 0.5 * lado
    else:  # de costas / rosto nao visivel: parte de cima do corpo
        larg, alt = x2 - x1, y2 - y1
        topo = min(alt * 0.40, larg * 1.1)
        rx1, rx2 = x1, x2
        ry1, ry2 = y1 - 0.05 * alt, y1 + topo
    rx1, ry1 = int(max(0, rx1)), int(max(0, ry1))
    rx2, ry2 = int(min(W, rx2)), int(min(H, ry2))
    if rx2 - rx1 < 16 or ry2 - ry1 < 16:
        return quadro[max(0, y1):y2, max(0, x1):x2], (x1, y1, x2, y2)
    return quadro[ry1:ry2, rx1:rx2], (rx1, ry1, rx2, ry2)


# ----------------------------------------------------------------------------
# Estado de cada pessoa
# ----------------------------------------------------------------------------
@dataclass
class EstadoEPI:
    votos: deque = field(default_factory=deque)      # (t, True/False)
    status: str = "?"                                 # "OK", "SEM", "?"
    sem_desde: float | None = None
    ultimo_alerta: float | None = None
    total_sem: float = 0.0
    ultimo_t: float | None = None


@dataclass
class Pessoa:
    chave: str                       # "P1" ou "T7" (rastreio sem identidade)
    primeiro: float
    ultimo: float
    presente: bool = True
    ausente_desde: float | None = None
    alertou_ausencia: bool = False
    total_presente: float = 0.0
    total_ausente: float = 0.0
    n_ausencias: int = 0
    epis: dict[str, EstadoEPI] = field(default_factory=dict)
    caixa: tuple = (0, 0, 0, 0)


@dataclass
class Rastreio:
    pid: str | None = None                   # identidade ja reconhecida
    desconhecidos: list = field(default_factory=list)   # vetores sem dono
    troca: tuple = (None, 0)                 # (outro id, quantas vezes seguidas)
    ultimo_rosto: float = -1e9
    caixa: tuple = (0, 0, 0, 0)
    visto: float = 0.0


# ----------------------------------------------------------------------------
# Registro em arquivo e alertas
# ----------------------------------------------------------------------------
class Registro:
    def __init__(self, cfg: dict, relogio: Relogio):
        self.cfg = cfg["alertas"]
        self.relogio = relogio
        self.pasta = (AQUI / cfg["pasta_dados"]).resolve()
        (self.pasta / "fotos").mkdir(parents=True, exist_ok=True)
        self.arq_alertas = self.pasta / "alertas.csv"
        self.arq_ausencias = self.pasta / "ausencias.csv"
        self.arq_resumo = self.pasta / "resumo_sessoes.csv"
        self.recentes: deque = deque(maxlen=4)   # mostrados na tela

    @staticmethod
    def _linha(arquivo: Path, cabecalho: list[str], valores: list):
        novo = not arquivo.exists()
        # utf-8-sig e ';' para abrir direto no Excel em portugues
        with arquivo.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f, delimiter=";")
            if novo:
                w.writerow(cabecalho)
            w.writerow(valores)

    def ausencia(self, pid, nome, t_ini, t_fim, encerrada_por="retorno"):
        dur = t_fim - t_ini
        self._linha(self.arq_ausencias,
                    ["id", "nome", "inicio", "fim", "duracao_s", "duracao_min", "duracao", "fim_por"],
                    [pid, nome, self.relogio.hora(t_ini).strftime("%d/%m/%Y %H:%M:%S"),
                     self.relogio.hora(t_fim).strftime("%d/%m/%Y %H:%M:%S"),
                     int(round(dur)), f"{dur / 60:.2f}".replace(".", ","),
                     fmt_duracao(dur), encerrada_por])

    def alerta(self, tipo: str, pid: str, nome: str, mensagem: str, quadro=None):
        agora = self.relogio.hora()
        foto = ""
        if quadro is not None and self.cfg.get("salvar_foto", True):
            foto = f"{agora:%Y%m%d_%H%M%S}_{tipo}_{pid}.jpg"
            cv2.imwrite(str(self.pasta / "fotos" / foto), quadro)
        self._linha(self.arq_alertas, ["data_hora", "tipo", "id", "nome", "mensagem", "foto"],
                    [agora.strftime("%d/%m/%Y %H:%M:%S"), tipo, pid, nome, mensagem, foto])
        texto = f"[ALERTA {agora:%H:%M:%S}] {mensagem}"
        print(texto, flush=True)
        self.recentes.append((self.relogio.t(), mensagem))
        if self.cfg.get("som", True):
            self._bipe()
        url = (self.cfg.get("webhook_url") or "").strip()
        if url:
            threading.Thread(target=self._webhook, args=(url, texto), daemon=True).start()
        if all(config_telegram(self.cfg)):
            arq_foto = (self.pasta / "fotos" / foto) if foto else None
            threading.Thread(target=enviar_telegram, args=(self.cfg, texto, arq_foto),
                             daemon=True).start()
        for tel, chave in destinos_whatsapp(self.cfg):
            threading.Thread(target=enviar_whatsapp, args=(tel, chave, texto),
                             daemon=True).start()

    @staticmethod
    def _bipe():
        try:
            import winsound
            winsound.Beep(1200, 350)
        except Exception:  # noqa: BLE001  (fora do Windows)
            print("\a", end="", flush=True)

    @staticmethod
    def _webhook(url: str, texto: str):
        try:
            dados = json.dumps({"text": texto, "content": texto}).encode("utf-8")
            req = urllib.request.Request(url, data=dados,
                                         headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=10).read()
        except Exception as e:  # noqa: BLE001
            print(f"  (falha ao enviar webhook: {e})")

    def resumo(self, pessoas: dict[str, Pessoa], galeria: Galeria | None, itens: list[str]):
        fim = self.relogio.hora().strftime("%d/%m/%Y %H:%M:%S")
        cab = (["sessao_fim", "id", "nome", "tempo_presente", "tempo_ausente",
                "presente_s", "ausente_s", "pct_presente", "n_ausencias"]
               + [f"sem_{i}_s" for i in itens])
        for p in pessoas.values():
            total = p.total_presente + p.total_ausente
            pct = 100 * p.total_presente / total if total > 0 else 0
            nome = galeria.nome(p.chave) if galeria else p.chave
            self._linha(self.arq_resumo, cab,
                        [fim, p.chave, nome, fmt_duracao(p.total_presente),
                         fmt_duracao(p.total_ausente), int(round(p.total_presente)),
                         int(round(p.total_ausente)), f"{pct:.1f}".replace(".", ","),
                         p.n_ausencias]
                        + [int(round(p.epis[i].total_sem)) if i in p.epis else 0 for i in itens])


# ----------------------------------------------------------------------------
# Nucleo: junta tudo
# ----------------------------------------------------------------------------
class Monitor:
    def __init__(self, cfg: dict, relogio: Relogio, det_pessoas, det_epi, rostos,
                 galeria: Galeria | None):
        self.cfg = cfg
        self.tp = cfg["tempos"]
        self.idc = cfg["identificacao"]
        self.relogio = relogio
        self.det_pessoas, self.det_epi, self.rostos = det_pessoas, det_epi, rostos
        self.galeria = galeria
        self.registro = Registro(cfg, relogio)
        self.itens: dict = cfg["epis"]
        self.rastreios: dict[int, Rastreio] = {}
        self.pessoas: dict[str, Pessoa] = {}
        self.perdidos: list[tuple[float, str, tuple]] = []   # (t, pid, caixa)
        self.ultimo_epi = -1e9
        self.ultimo_t: float | None = None

    # ---------------- identidade ----------------
    def _nome(self, chave: str) -> str:
        if chave.startswith("T"):
            return f"Nao identificado ({chave})"
        return self.galeria.nome(chave) if self.galeria else chave

    def _dono_atual(self, pid: str, menos: int) -> int | None:
        for tid, r in self.rastreios.items():
            if tid != menos and r.pid == pid:
                return tid
        return None

    def _identificar(self, quadro, tid: int, r: Rastreio, face, t: float):
        if face is None or self.rostos is None:
            return
        if min(face[2], face[3]) < self.idc["tamanho_minimo_rosto"] or face[14] < 0.8:
            return
        # com identidade: reconfere a cada 2 s; sem identidade: a cada quadro
        if r.pid and t - r.ultimo_rosto < 2.0:
            return
        r.ultimo_rosto = t
        vetor = self.rostos.vetor(quadro, face)
        pid, nota = self.galeria.comparar(vetor)

        if pid and nota >= self.idc["limiar_mesma_pessoa"]:
            if r.pid is None:
                outro = self._dono_atual(pid, tid)
                if outro is not None:          # outra pessoa ja esta com este ID agora
                    return
                r.pid, r.desconhecidos = pid, []
                self.galeria.reforcar(pid, vetor, nota)
            elif r.pid == pid:
                r.troca = (None, 0)
                self.galeria.reforcar(pid, vetor, nota)
            else:                              # rastreador trocou as pessoas?
                cand, n = r.troca
                r.troca = (pid, n + 1) if cand == pid else (pid, 1)
                if r.troca[1] >= 3 and nota >= 0.5 and self._dono_atual(pid, tid) is None:
                    r.pid, r.troca = pid, (None, 0)
        elif r.pid is None and nota >= self.idc["limiar_pessoa_nova"]:
            r.desconhecidos = []               # parecido com alguem: nao cadastra como novo
        elif r.pid is None:
            r.desconhecidos.append(vetor)
            if len(r.desconhecidos) >= self.idc["leituras_para_cadastrar"]:
                r.pid = self.galeria.nova(r.desconhecidos, self.relogio.hora())
                r.desconhecidos = []
                print(f"Nova pessoa cadastrada: {r.pid}", flush=True)

    # ---------------- EPI ----------------
    def _votar_epi(self, p: Pessoa, achados: dict, tem_rosto: bool, t: float, quadro):
        for nome_item, item in self.itens.items():
            e = p.epis.setdefault(nome_item, EstadoEPI())
            if item.get("precisa_rosto") and not tem_rosto:
                e.ultimo_t = None             # nao da para avaliar: nao conta tempo
                continue
            e.votos.append((t, nome_item in achados))
            while e.votos and t - e.votos[0][0] > self.tp["janela_epi"]:
                e.votos.popleft()
            cobertura = t - e.votos[0][0]
            if len(e.votos) >= 3 and cobertura >= self.tp["janela_epi"] * 0.6:
                positivos = sum(1 for _, v in e.votos if v)
                e.status = "OK" if positivos / len(e.votos) >= 0.5 else "SEM"
            if e.status == "SEM":
                if e.ultimo_t is not None:
                    e.total_sem += t - e.ultimo_t
                if e.sem_desde is None:
                    e.sem_desde = t
                tempo_sem = t - e.sem_desde
                pode = (e.ultimo_alerta is None
                        or t - e.ultimo_alerta >= self.tp["repetir_alerta_epi"])
                if item.get("obrigatorio", True) and pode \
                        and tempo_sem >= self.tp["sem_epi_para_alerta"]:
                    e.ultimo_alerta = t
                    rotulo = nome_item.replace("_", " ")
                    self.registro.alerta(
                        "SEM_EPI", p.chave, self._nome(p.chave),
                        f"{self._nome(p.chave)} sem {rotulo} ha {fmt_duracao(tempo_sem)}",
                        quadro)
            elif e.status == "OK":
                e.sem_desde, e.ultimo_alerta = None, None
            e.ultimo_t = t

    # ---------------- presenca ----------------
    def _marcar_presente(self, chave: str, caixa, t: float) -> Pessoa:
        p = self.pessoas.get(chave)
        if p is None:
            p = self.pessoas[chave] = Pessoa(chave=chave, primeiro=t, ultimo=t)
        if not p.presente:                    # voltou
            self.registro.ausencia(chave, self._nome(chave), p.ausente_desde, t)
            print(f"{self._nome(chave)} voltou apos {fmt_duracao(t - p.ausente_desde)} ausente",
                  flush=True)
            p.presente, p.ausente_desde, p.alertou_ausencia = True, None, False
            for e in p.epis.values():         # recomeca a avaliacao de EPI
                e.votos.clear()
                e.status, e.sem_desde, e.ultimo_t = "?", None, None
        p.ultimo, p.caixa = t, caixa
        return p

    def _atualizar_ausencias(self, vistos: set[str], t: float, dt: float, quadro):
        for chave, p in list(self.pessoas.items()):
            if chave.startswith("T"):         # sem identidade: nao da para medir ausencia
                if chave not in vistos and t - p.ultimo > self.tp["tolerancia_ausencia"]:
                    del self.pessoas[chave]
                continue
            if chave in vistos:
                p.total_presente += dt
                continue
            if p.presente:
                if t - p.ultimo > self.tp["tolerancia_ausencia"]:
                    p.presente, p.ausente_desde = False, p.ultimo
                    p.n_ausencias += 1
                    # o tempo de tolerancia ja passado vira ausencia
                    passado = t - p.ultimo
                    p.total_ausente += passado
                    p.total_presente -= min(p.total_presente, passado - dt)
                else:
                    p.total_presente += dt
            else:
                p.total_ausente += dt
                fora = t - p.ausente_desde
                if not p.alertou_ausencia and fora >= self.tp["alerta_ausencia"]:
                    p.alertou_ausencia = True
                    self.registro.alerta(
                        "AUSENCIA", chave, self._nome(chave),
                        f"{self._nome(chave)} ausente ha {fmt_duracao(fora)}", quadro)

    # ---------------- um quadro ----------------
    def processar(self, quadro) -> np.ndarray:
        t = self.relogio.t()
        dt = 0.0 if self.ultimo_t is None else max(0.0, t - self.ultimo_t)
        self.ultimo_t = t

        deteccoes = self.det_pessoas.rastrear(quadro)
        faces = self.rostos.detectar(quadro) if self.rostos else []

        # associa cada rosto a pessoa cuja caixa contem o centro do rosto
        face_de: dict[int, np.ndarray] = {}
        for f in faces:
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            melhor, menor_area = None, None
            for tid, (x1, y1, x2, y2) in deteccoes:
                if x1 <= cx <= x2 and y1 <= cy <= y1 + (y2 - y1) * 0.7:
                    area = (x2 - x1) * (y2 - y1)
                    if menor_area is None or area < menor_area:
                        melhor, menor_area = tid, area
            if melhor is not None and (melhor not in face_de or f[14] > face_de[melhor][14]):
                face_de[melhor] = f

        # rastreios que sumiram: guarda a identidade por alguns segundos
        ativos = {tid for tid, _ in deteccoes}
        for tid in [k for k in self.rastreios if k not in ativos]:
            r = self.rastreios[tid]
            if t - r.visto > 1.0:
                if r.pid:
                    self.perdidos.append((r.visto, r.pid, r.caixa))
                del self.rastreios[tid]
        self.perdidos = [x for x in self.perdidos if t - x[0] <= 4.0]

        vistos: set[str] = set()
        linhas = []      # (pessoa, caixa, face)
        for tid, caixa in deteccoes:
            r = self.rastreios.get(tid)
            if r is None:
                r = self.rastreios[tid] = Rastreio()
                # o rastreador perdeu e reencontrou a mesma pessoa no mesmo lugar?
                for i, (_, pid, cx_antiga) in enumerate(self.perdidos):
                    if iou(caixa, cx_antiga) > 0.4 and self._dono_atual(pid, tid) is None:
                        r.pid = pid
                        del self.perdidos[i]
                        break
            r.caixa, r.visto = caixa, t
            face = face_de.get(tid)
            if self.galeria is not None:
                self._identificar(quadro, tid, r, face, t)
                chave = r.pid or f"T{tid}"
            else:
                chave = f"ID{tid}"           # modo sem rosto: ID do rastreador
            if r.pid and f"T{tid}" in self.pessoas:    # acabou de ganhar identidade
                antigo = self.pessoas.pop(f"T{tid}")
                if r.pid not in self.pessoas:
                    antigo.chave = r.pid
                    self.pessoas[r.pid] = antigo
            p = self._marcar_presente(chave, caixa, t)
            vistos.add(chave)
            linhas.append((p, caixa, face))

        # EPIs (nao precisa ser em todo quadro)
        if linhas and t - self.ultimo_epi >= self.tp["intervalo_epi"]:
            self.ultimo_epi = t
            recortes = [recorte_cabeca(quadro, cx, f)[0] for _, cx, f in linhas]
            for (p, _, f), achados in zip(linhas, self.det_epi.conferir(recortes)):
                self._votar_epi(p, achados, f is not None, t, quadro)

        self._atualizar_ausencias(vistos, t, dt, quadro)
        return self._desenhar(quadro.copy(), linhas, t)

    # ---------------- tela ----------------
    def _desenhar(self, img, linhas, t: float):
        fonte = cv2.FONT_HERSHEY_SIMPLEX
        for p, (x1, y1, x2, y2), _ in linhas:
            falta = [n for n, it in self.itens.items()
                     if it.get("obrigatorio", True) and p.epis.get(n) and p.epis[n].status == "SEM"]
            cor = VERMELHO if falta else VERDE
            cv2.rectangle(img, (x1, y1), (x2, y2), cor, 2)
            titulo = self._nome(p.chave)
            if not p.chave.startswith("T") and self._nome(p.chave) != p.chave:
                titulo = f"{p.chave} {titulo}"
            (tw, th), _ = cv2.getTextSize(titulo, fonte, 0.6, 2)
            cv2.rectangle(img, (x1, max(0, y1 - th - 10)), (x1 + tw + 8, y1), cor, -1)
            cv2.putText(img, titulo, (x1 + 4, max(th + 2, y1 - 6)), fonte, 0.6, BRANCO, 2)
            y = y1 + 22
            for n in self.itens:
                st = p.epis[n].status if n in p.epis else "?"
                c = {"OK": VERDE, "SEM": VERMELHO}.get(st, CINZA)
                txt = f"{n.replace('_', ' ').upper()}: {st}"
                (tw, th), _ = cv2.getTextSize(txt, fonte, 0.5, 1)
                cv2.rectangle(img, (x1 + 3, y - th - 4), (x1 + tw + 10, y + 5), (25, 25, 25), -1)
                cv2.putText(img, txt, (x1 + 6, y), fonte, 0.5, c, 1)
                y += 20

        # painel de presenca
        fixos = [p for k, p in sorted(self.pessoas.items()) if not k.startswith("T")]
        alt = 30 + 22 * max(1, len(fixos))
        painel = img[0:alt, 0:430].copy()
        cv2.rectangle(img, (0, 0), (430, alt), (30, 30, 30), -1)
        cv2.addWeighted(painel, 0.25, img[0:alt, 0:430], 0.75, 0, img[0:alt, 0:430])
        cv2.putText(img, f"PRESENCA   {self.relogio.hora():%H:%M:%S}", (8, 20),
                    fonte, 0.55, BRANCO, 1)
        y = 42
        for p in fixos:
            if p.presente:
                txt, c = f"{self._nome(p.chave)}: presente {fmt_duracao(p.total_presente)}", VERDE
            else:
                fora = t - p.ausente_desde
                c = VERMELHO if fora >= self.tp["alerta_ausencia"] else AMARELO
                txt = f"{self._nome(p.chave)}: AUSENTE ha {fmt_duracao(fora)}"
            cv2.putText(img, txt, (8, y), fonte, 0.5, c, 1)
            y += 22
        if not fixos:
            cv2.putText(img, "aguardando reconhecer alguem...", (8, y), fonte, 0.5, CINZA, 1)

        # ultimos alertas no rodape
        H = img.shape[0]
        y = H - 12
        for quando, msg in reversed(self.registro.recentes):
            if t - quando > 15:
                continue
            (tw, th), _ = cv2.getTextSize(msg, fonte, 0.6, 2)
            cv2.rectangle(img, (0, y - th - 8), (tw + 16, y + 6), VERMELHO, -1)
            cv2.putText(img, msg, (8, y), fonte, 0.6, BRANCO, 2)
            y -= th + 18
        return img

    # ---------------- fim ----------------
    def encerrar(self):
        t = self.relogio.t()
        for chave, p in self.pessoas.items():
            if not chave.startswith("T") and not p.presente:
                self.registro.ausencia(chave, self._nome(chave), p.ausente_desde, t,
                                       "fim_da_sessao")
        fixos = {k: p for k, p in self.pessoas.items() if not k.startswith("T")}
        self.registro.resumo(fixos, self.galeria, list(self.itens))
        print("\n===== RESUMO DA SESSAO =====")
        for k, p in fixos.items():
            total = p.total_presente + p.total_ausente
            pct = 100 * p.total_presente / total if total else 0
            print(f"{self._nome(k):<22} presente {fmt_duracao(p.total_presente):>14} | "
                  f"ausente {fmt_duracao(p.total_ausente):>14} | {pct:5.1f}% | "
                  f"{p.n_ausencias} ausencia(s)")
            for n, e in p.epis.items():
                if e.total_sem >= 1:
                    print(f"{'':<22}   sem {n.replace('_', ' ')}: {fmt_duracao(e.total_sem)}")
        print(f"Arquivos em: {self.registro.pasta}")


# ----------------------------------------------------------------------------
# Programa principal
# ----------------------------------------------------------------------------
def carregar_config(caminho: Path) -> dict:
    if not caminho.exists():
        exemplo = caminho.with_name("config.example.yaml")
        if caminho.name == "config.yaml" and exemplo.exists():
            caminho.write_text(exemplo.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"Criei {caminho.name} a partir de {exemplo.name}. "
                  "Edite-o para ajustar camera e alertas.")
        else:
            raise SystemExit(f"Nao achei o arquivo de configuracao: {caminho}")
    return yaml.safe_load(caminho.read_text(encoding="utf-8"))


def abrir_video(fonte):
    if isinstance(fonte, int) and sys.platform.startswith("win"):
        cap = cv2.VideoCapture(fonte, cv2.CAP_DSHOW)   # abre mais rapido no Windows
    else:
        cap = cv2.VideoCapture(fonte)
    if not cap.isOpened():
        raise SystemExit(f"Nao consegui abrir a camera/video: {fonte}\n"
                         "Feche Teams/Zoom/app Camera ou troque 'camera' no config.yaml.")
    return cap


def main():
    ap = argparse.ArgumentParser(description="Monitor de EPI e presenca")
    ap.add_argument("--config", default=str(AQUI / "config.yaml"))
    ap.add_argument("--video", help="arquivo de video para testar (em vez da camera)")
    ap.add_argument("--sem-janela", action="store_true", help="roda sem mostrar a imagem")
    ap.add_argument("--gravar", help="salva o video com as marcacoes neste arquivo .mp4")
    ap.add_argument("--diagnostico", action="store_true",
                    help="abre uma 2a janela com tudo que o detector de EPI encontra")
    ap.add_argument("--listar-cameras", action="store_true",
                    help="testa as cameras 0 a 5 e salva uma foto de cada em dados/")
    ap.add_argument("--testar-whatsapp", action="store_true",
                    help="envia uma mensagem de teste aos numeros do config.yaml")
    ap.add_argument("--testar-telegram", action="store_true",
                    help="descobre o chat_id e envia uma mensagem de teste pelo Telegram")
    ap.add_argument("--listar", action="store_true", help="lista pessoas cadastradas")
    ap.add_argument("--renomear", nargs=2, metavar=("ID", "NOME"))
    ap.add_argument("--apagar", metavar="ID", help="apaga uma pessoa do cadastro")
    args = ap.parse_args()

    cfg = carregar_config(Path(args.config))
    arq_galeria = (AQUI / cfg["pasta_dados"] / "pessoas.json")

    if args.listar_cameras:
        pasta = (AQUI / cfg["pasta_dados"]).resolve()
        pasta.mkdir(parents=True, exist_ok=True)
        achou = False
        for i in range(6):
            if sys.platform.startswith("win"):
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            ok, quadro = (cap.read() if cap.isOpened() else (False, None))
            cap.release()
            if ok and quadro is not None:
                achou = True
                arq = pasta / f"camera_{i}.jpg"
                cv2.imwrite(str(arq), quadro)
                print(f"camera: {i}   {quadro.shape[1]}x{quadro.shape[0]}   foto: {arq}")
        if not achou:
            print("Nenhuma camera encontrada (feche Teams/Zoom/app Camera e tente de novo).")
        else:
            print("Abra as fotos para ver qual e qual e coloque o numero em 'camera' no config.yaml.")
        return

    if args.testar_telegram:
        token, chat = config_telegram(cfg["alertas"])
        if not token:
            raise SystemExit("Preencha o token em alertas > telegram no config.yaml.")
        if not chat:
            r = telegram_chamar(token, "getUpdates", {})
            if not r.get("ok"):
                raise SystemExit(f"O Telegram recusou o token: {r.get('description')}")
            chats = {}
            for u in r.get("result", []):
                for chave in ("message", "my_chat_member", "channel_post"):
                    c = (u.get(chave) or {}).get("chat")
                    if c:
                        chats[c["id"]] = c.get("title") or c.get("first_name") or c.get("username") or ""
            if not chats:
                raise SystemExit("Token OK, mas o bot ainda nao recebeu nenhuma mensagem.\n"
                                 "No Telegram, abra a conversa com o seu bot, mande 'oi' e rode de novo.")
            print("Token OK. Conversas encontradas:")
            for cid, nome in chats.items():
                print(f"   chat_id: {cid}    ({nome})")
            print("Copie o chat_id desejado para o config.yaml (alertas > telegram > chat_id) e rode de novo.")
            return
        ok = enviar_telegram(cfg["alertas"], "Teste do Monitor de EPI: o alerta por Telegram esta funcionando.")
        print("Mensagem de teste enviada." if ok else "NAO enviada.")
        return

    if args.testar_whatsapp:
        destinos = destinos_whatsapp(cfg["alertas"])
        if not destinos:
            raise SystemExit("Preencha telefone e apikey em alertas > whatsapp no config.yaml.")
        for tel, chave in destinos:
            ok = enviar_whatsapp(tel, chave, "Teste do Monitor de EPI: o alerta por WhatsApp esta funcionando.")
            print(f"{tel}: {'enviado' if ok else 'NAO enviado'}")
        return

    if args.listar or args.renomear or args.apagar:
        g = Galeria(arq_galeria)
        if args.renomear:
            pid, nome = args.renomear
            if pid not in g.pessoas:
                raise SystemExit(f"ID {pid} nao existe.")
            g.pessoas[pid]["nome"] = nome
            g.salvar()
        if args.apagar:
            if g.pessoas.pop(args.apagar, None) is None:
                raise SystemExit(f"ID {args.apagar} nao existe.")
            g.salvar()
        for pid, p in g.pessoas.items():
            print(f"{pid:<5} {p['nome'] or '(sem nome)':<25} cadastrado em {p['criado']}")
        if not g.pessoas:
            print("Nenhuma pessoa cadastrada ainda.")
        return

    fonte = args.video if args.video else cfg["camera"]
    cap = abrir_video(fonte)
    if not args.video and isinstance(fonte, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.get("largura", 1280))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.get("altura", 720))
    fps_arquivo = (cap.get(cv2.CAP_PROP_FPS) or 30.0) if args.video else None
    relogio = Relogio(fps_arquivo)

    print("Carregando modelos (na primeira vez baixa da internet, aguarde)...")
    usar_rosto = cfg["identificacao"].get("usar_rosto", True)
    rostos = Rostos(cfg["identificacao"]) if usar_rosto else None
    galeria = Galeria(arq_galeria) if usar_rosto else None
    monitor = Monitor(cfg, relogio, DetectorPessoas(cfg), DetectorEPI(cfg), rostos, galeria)
    monitor.det_epi.diagnostico = args.diagnostico
    print("Pronto. Pressione Q na janela para sair.")

    gravador = None
    try:
        while True:
            ok, quadro = cap.read()
            if not ok:
                if args.video:
                    break
                time.sleep(0.2)               # camera IP caiu? tenta de novo
                continue
            relogio.avancar()
            saida = monitor.processar(quadro)
            if args.gravar:
                if gravador is None:
                    h, w = saida.shape[:2]
                    gravador = cv2.VideoWriter(args.gravar, cv2.VideoWriter_fourcc(*"mp4v"),
                                               fps_arquivo or 10, (w, h))
                gravador.write(saida)
            if not args.sem_janela:
                cv2.imshow("Monitor de EPI e Presenca (Q = sair)", saida)
                diag = monitor.det_epi.imagem_diag
                if args.diagnostico and diag is not None:
                    esc = 520 / diag.shape[0]
                    cv2.imshow("Diagnostico EPI", cv2.resize(diag, None, fx=esc, fy=esc))
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        monitor.encerrar()
        cap.release()
        if gravador is not None:
            gravador.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
