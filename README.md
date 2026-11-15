# Estratégias de Inicialização em Redes Neurais baseadas em Hiperretângulos

Trabalho de Conclusão de Curso (USP, Ribeirão Preto), orientado pelo Prof. Denis Mayr Lima Martins.

Este repositório estende a implementação da **HyperNN** ([mlde-ms/hypernn](https://github.com/mlde-ms/hypernn), licença MIT), uma rede neural diferenciável para classificação com hiperretângulos (caixas), cujas decisões são regras interpretáveis. No original, as caixas iniciais são sorteadas de uma normal padrão. Aqui se investiga se inicializá-las a partir dos dados melhora o resultado.

## O que foi investigado

- **Inicializações:** SOM, K-Means, DBSCAN, Random (original) e Xavier, com busca em dois estágios (estágio 1: escolha da inicialização e do número de caixas M sem treinar; estágio 2: busca da taxa de aprendizado por validação cruzada).
- **Aquecimento da regularização:** os coeficientes L1/L2 crescem gradualmente nas primeiras W épocas. Sem ele, caixas iniciais pequenas colapsam sob regularização.
- **Bases:** Iris, Wine, Breast Cancer, Digits e Vehicle, em classificação um-contra-todos, com 3 sementes.

## Principais conclusões

- Nenhuma inicialização guiada por dados traz ganho significativo sobre a inicialização aleatória.
- O resultado mais relevante é o aquecimento da regularização, que elimina o colapso.
- O texto completo, com metodologia e testes estatísticos, está na monografia (link/DOI quando disponível).

## Estrutura

- `run_experiments.py`: executa os experimentos.
- `src/models/`: HyperNN (`hypernn.py`, `hypernnclassifier.py`) e as inicializações (`hypernnwithinitclassifier.py`).
- `src/evaluation/`: busca de hiperparâmetros, avaliação e figuras.
- `src/dataset/`: carregamento das bases.
- `results/`: resultados brutos (CSV).

## Como executar

```bash
pip install -r requirements.txt
python run_experiments.py --init_strategy all
```

Opções úteis: `--init_strategy {som,kmeans,dbscan,random,xavier,all}`, `--max_train_epochs`, `--patience`, `--num_seeds`, `--use_cuda`.

O `requirements.txt` instala o PyTorch com CUDA 12.1. Para rodar sem GPU, troque `torch==2.5.1+cu121` por `torch==2.5.1` e apague a linha `--extra-index-url`.

O arquivo `experiment_checkpoint.json` guarda resultados parciais. Apague-o para refazer uma execução do zero.

## Créditos e licença

Baseado em:
> MARTINS, D. M. L.; LÜLF, C.; GIESEKE, F. End-to-End Neural Network Training for Hyperbox-Based Classification. ESANN, 2023.

Distribuído sob a licença MIT, mantendo o aviso de copyright do projeto original.
