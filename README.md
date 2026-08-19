<div align="center">

# 🪐 ExoHunter-ML
### *Exoplanet Transit Detection via Machine Learning & Flux Feature Extraction*

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg?logo=jupyter&logoColor=white)](https://jupyter.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7%2B-red.svg)](https://xgboost.readthedocs.io/)
[![NASA Kepler](https://img.shields.io/badge/Data-NASA%20Kepler%20Space%20Telescope-11244D?logo=nasa&logoColor=white)](https://exoplanetarchive.ipac.caltech.edu/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<br/>

> **Uma pipeline completa de Machine Learning aplicada à Astrofísica Observacional para identificação automatizada de exoplanetas a partir de curvas de luz e variações no fluxo estelar (Fotometria de Trânsito).**

</div>

---

## 📌 Sumário
- [Visão Geral](#-visão-geral)
- [Fundamentação Física & Modelagem Matemática](#-fundamentação-física--modelagem-matemática)
- [Arquitetura da Pipeline](#-arquitetura-da-pipeline)
- [Engenharia de Features](#-engenharia-de-features)
- [Resultados & Desempenho](#-resultados--desempenho)
- [Estrutura do Repositório](#-estrutura-do-repositório)
- [Como Executar](#-como-executar)
- [Autor](#-autor)

---

## 🔭 Visão Geral

O **ExoHunter-ML** aborda o desafio astronômico de classificar e confirmar candidatas a exoplanetas analisando séries temporais do **Telescópio Espacial Kepler (NASA)**. 

Devido ao desbalanceamento extremo de classes presente em observações astronômicas reais (onde $99\%+$ dos sinais observados pertencem a estrelas sem trânsitos detectáveis ou falsos-positivos astrofísicos), este projeto aplica técnicas avançadas de:
1. **Tratamento de Desbalanceamento de Dados** (SMOTE / Undersampling / Pesos de Classe).
2. **Engenharia de Features Físicas e Estatísticas** sobre sinais fotométricos de fluxo ($FLUX_1 \dots FLUX_n$).
3. **Modelagem Preditiva Ensembled** (Random Forest, XGBoost, Decision Trees e Classificadores Lineares).

---

## 📐 Fundamentação Física & Modelagem Matemática

O projeto fundamenta-se na **Fotometria de Trânsito**, na qual o fluxo luminoso aparente de uma estrela hospedeira sofre um decréscimo periódico quando um exoplaneta cruza o disco estelar em relação ao campo de visão do observador.