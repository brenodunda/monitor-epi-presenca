# Monitor de EPI e Presença

Visão computacional em Python para postos de trabalho industriais. Com uma câmera comum, o programa:

- identifica cada pessoa na imagem e dá a ela um **ID fixo** (P1, P2…) pelo rosto, mesmo que saia e volte;
- confere **capacete, óculos de segurança e protetor auricular** de cada pessoa;
- mede o **tempo presente e o tempo ausente** de cada ID, em segundos e minutos;
- emite **alertas** quando alguém fica sem EPI ou ausente além de um limite: na tela, com bipe, em CSV, com foto e, opcionalmente, por **Telegram**, **WhatsApp** ou webhook.

Roda em notebook comum, sem placa de vídeo, com webcam USB ou câmera IP (RTSP).

> **Estado do projeto:** protótipo validado em teste de mesa, com uma pessoa e uma webcam. Ainda não foi validado em chão de fábrica.

## Como funciona

```
câmera ─► YOLO11 (pessoas) + ByteTrack (rastreio)
            │
            ├─► YuNet + SFace (rosto) ─► ID fixo por pessoa (P1, P2…)
            │
            └─► recorte da cabeça ─► YOLOE (capacete, óculos, abafador)
                                     └─► classificador treinado (abafador na orelha?)
                                            │
                     votação em janela de 3 s ─► status OK / SEM por EPI
                                            │
            tempo presente / ausente por ID ─► alertas + CSV + foto
```

- **Pessoas:** YOLO11n detecta e o ByteTrack rastreia entre quadros.
- **Identidade:** o rosto é convertido em um vetor numérico (SFace) e comparado ao cadastro. Pessoas novas são cadastradas sozinhas. O cadastro guarda só os vetores, não as fotos.
- **EPIs:** o YOLOE é um detector de "vocabulário aberto": acha objetos pela descrição em inglês ("hard hat", "safety glasses"…), sem treino. Para o abafador há um classificador próprio, treinado com fotos tiradas pela própria câmera (veja [Treinar o abafador](#treinar-o-abafador)).
- **Decisão:** cada EPI é avaliado a cada 0,5 s e o status vem da maioria das leituras dos últimos 3 s, o que evita alarme por uma leitura errada.

## Instalação (Windows)

Precisa do [Python](https://www.python.org/downloads/) 3.10 ou mais novo, instalado com a opção "Add Python to PATH".

Baixe o repositório (botão **Code → Download ZIP**, ou `git clone`), abra o Prompt de Comando dentro da pasta e rode:

```
python -m venv venv
venv\Scripts\python -m pip install --upgrade pip
venv\Scripts\python -m pip install -r requirements.txt
```

A instalação baixa cerca de 1 GB (PyTorch). Na primeira execução o programa baixa os modelos (cerca de 700 MB) e cria o `config.yaml` a partir do `config.example.yaml`.

No Linux ou macOS, troque `venv\Scripts\python` por `venv/bin/python`.

## Uso

| Comando | O que faz |
|---|---|
| `venv\Scripts\python monitor_epi.py` | Abre a câmera e começa a monitorar |
| `... monitor_epi.py --diagnostico` | Abre uma 2ª janela com tudo que o detector de EPI encontra e a nota de cada item |
| `... monitor_epi.py --listar-cameras` | Testa as câmeras 0 a 5 e salva uma foto de cada em `dados/`, para saber o número de cada uma |
| `... monitor_epi.py --video arquivo.mp4` | Roda sobre um vídeo gravado |
| `... monitor_epi.py --gravar saida.mp4` | Grava o vídeo com as marcações |
| `... monitor_epi.py --listar` | Lista as pessoas cadastradas |
| `... monitor_epi.py --renomear P1 "Maria"` | Dá um nome a um ID |
| `... monitor_epi.py --apagar P3` | Apaga um ID do cadastro |
| `... monitor_epi.py --testar-telegram` | Mostra o `chat_id` e envia uma mensagem de teste |
| `... monitor_epi.py --testar-whatsapp` | Envia uma mensagem de teste pelo WhatsApp |

Na janela, **Q** ou **Esc** encerra e grava o resumo da sessão.

### Teste rápido na mesa

O `config.example.yaml` vem com o alerta de ausência em 60 s, para dar para testar sentado.

1. Fique de frente para a câmera: em poucos segundos aparece "Nova pessoa cadastrada: P1".
2. Coloque os EPIs: os três ficam **OK** em verde.
3. Tire o capacete: em cerca de 12 s toca o bipe e aparece o alerta.
4. Saia da imagem por mais de 1 minuto: alerta de ausência.
5. Volte: ele deve reconhecer você como P1 de novo e registrar a ausência.

## Configuração

Tudo fica no `config.yaml` (crie a partir do `config.example.yaml`; ele não vai para o Git porque guarda token e telefone). Os ajustes mais comuns:

| Situação | O que mudar |
|---|---|
| Abriu a câmera errada | `camera:` (use `--listar-cameras` para descobrir o número) |
| Câmera IP | `camera: "rtsp://usuario:senha@192.168.0.50:554/stream1"` |
| Diz OK sem o EPI | aumente `confianca` daquele EPI |
| Não reconhece o EPI | diminua `confianca` ou aumente `tamanho_imagem_epi` para 640 |
| Está lento | `tamanho_imagem_epi: 320` e `intervalo_epi: 1.0` |
| Criou dois IDs para a mesma pessoa | apague o duplicado com `--apagar` e diminua `limiar_mesma_pessoa` |
| Alerta de ausência de 1 hora | `alerta_ausencia: 3600` |
| Não quer reconhecimento facial | `usar_rosto: false` (o ID passa a ser o do rastreador e muda quando a pessoa sai e volta) |

## Treinar o abafador

O detector genérico reconhece bem o abafador de lado, mas não de frente, e não distingue concha **na orelha** de concha **levantada** (presa no capacete). Por isso há um classificador próprio, treinado em poucos minutos no próprio notebook:

```
venv\Scripts\python capturar_fotos.py
venv\Scripts\python treinar_abafador.py
```

No `capturar_fotos.py`, grave as três situações com as teclas **1** (na orelha), **2** (levantado) e **3** (sem abafador), mexendo a cabeça devagar em todas as direções até passar de 150 fotos em cada uma. **Espaço** pausa e **Q** encerra. O treino grava `modelos/abafador_cls.pt`, que o monitor passa a usar sozinho; só "na orelha" conta como OK.

Para uso real, capture com vários operadores, várias cores de capacete e no posto de trabalho, com a mesma câmera que vai ficar instalada.

## Alertas

Todo alerta vai para a tela, para `dados/alertas.csv` e tem foto em `dados/fotos/`. Além disso:

- **Telegram (recomendado):** gratuito, oficial, envia a foto junto e funciona em grupo. Crie um bot com o [@BotFather](https://t.me/BotFather), coloque o token em `alertas > telegram > token`, mande uma mensagem ao bot (ou adicione-o a um grupo) e rode `--testar-telegram` para descobrir o `chat_id`.
- **WhatsApp via [CallMeBot](https://www.callmebot.com/):** gratuito, só texto, só para uso pessoal e com vagas limitadas. Serve para teste. Para uso em empresa, o caminho é a API oficial do WhatsApp Business.
- **Webhook:** `alertas > webhook_url` recebe um POST JSON com o texto do alerta (Teams, Slack, Discord e similares).

Os alertas de EPI se repetem a cada `repetir_alerta_epi` segundos enquanto o EPI continuar faltando; o de ausência sai uma vez por ausência.

## Arquivos gerados (`dados/`)

| Arquivo | Conteúdo |
|---|---|
| `ausencias.csv` | cada ausência: ID, nome, início, fim, duração em s e em min |
| `alertas.csv` | cada alerta, com data, hora, tipo e nome da foto |
| `resumo_sessoes.csv` | por pessoa e sessão: tempo presente, ausente, % de presença e tempo sem cada EPI |
| `fotos/` | imagem do momento de cada alerta |
| `pessoas.json` | cadastro dos IDs (vetores numéricos do rosto) |

Os CSV usam `;` e UTF-8 com BOM, então abrem direto no Excel em português e servem de fonte para o Power BI.

## Limitações conhecidas

- **Óculos:** não distingue óculos de grau de óculos de segurança.
- **Plug de ouvido:** pequeno demais para câmera a mais de ~1 m. O abafador tipo concha funciona bem melhor.
- **Ausente não é improdutivo:** o sistema mede "fora do campo da câmera". Buscar material ou ajudar em outro posto conta como ausência.
- **De costas:** capacete e abafador continuam sendo avaliados; óculos fica "?". A identidade se mantém enquanto o rastreio não se perde.
- **Um computador sem GPU** aguenta uma câmera. Para vários postos, use um PC com placa de vídeo NVIDIA.

## Privacidade e LGPD

Reconhecimento facial trata **dado biométrico**, que a LGPD classifica como dado pessoal sensível. Antes de usar com pessoas reais em uma empresa:

- alinhe com RH, jurídico e o encarregado de dados (DPO): base legal, finalidade, aviso aos colaboradores, quem acessa e por quanto tempo os dados ficam guardados;
- nunca publique as pastas `dados/` e `dataset_abafador/`, que têm rostos e biometria (o `.gitignore` já as exclui);
- se não puder usar biometria, use `usar_rosto: false` e meça por **posto** (ocupado ou vazio) em vez de por pessoa.

## Licença

[AGPL-3.0](LICENSE), a mesma da biblioteca [Ultralytics](https://github.com/ultralytics/ultralytics), da qual o projeto depende. Uso comercial da Ultralytics pode exigir a licença Enterprise deles.

## Créditos

- [Ultralytics YOLO11 e YOLOE](https://github.com/ultralytics/ultralytics): detecção de pessoas e de EPI, rastreio ByteTrack e treino do classificador
- [OpenCV Zoo](https://github.com/opencv/opencv_zoo): YuNet (detecção de rosto) e SFace (reconhecimento de rosto)
