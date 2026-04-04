# 🤖 BipedalWalker-v3: Um Estudo de Caso em PPO Contínuo

> **Status:** Concluído ✅  
> **Algoritmo:** PPO (Proximal Policy Optimization)  
> **Arquitetura:** Actor-Critic (On-Policy)  
> **Ambiente:** BipedalWalker-v3 (Controle Contínuo)

Este repositório contém uma implementação feita **do zero** do algoritmo **Proximal Policy Optimization (PPO)** com arquitetura **Actor-Critic**, aplicado a um dos ambientes mais instáveis e desafiadores do Gymnasium: o `BipedalWalker-v3` (Controle Contínuo).

O objetivo deste projeto não foi apenas "resolver" o ambiente usando bibliotecas prontas, mas sim construir a matemática do gradiente, a função de Clipping e a Distribuição Gaussiana de ações **do zero**, para observar e documentar o comportamento intrínseco de uma Inteligência Artificial lidando com um espaço de estados de alta dimensão.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep_Learning-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Gymnasium](https://img.shields.io/badge/Gymnasium-BipedalWalker-lightgrey?style=for-the-badge)
![RL](https://img.shields.io/badge/RL-PPO-green?style=for-the-badge)

---

## 🗂️ Estrutura do Projeto

```text
bipedal-walker-ppo/
│
├── checkpoints/           # Pesos do modelo treinado (.pth)
│
├── assets/                # Vídeos de avaliação (.mp4)
│
├── src/
│   ├── agent.py          # Implementação PPO do zero
│   └── model.py          # Arquitetura Actor-Critic
│
├── train.py              # Loop de treinamento (10k episódios)
├── evaluate.py           # Visualização e geração de vídeo
│
├── requirements.txt      # Dependências (requer swig + box2d)
│
└── README.md            # Este arquivo
```

---

## 🧠 Arquitetura do Agente

### 🔹 Componentes Principais

**Algoritmo:** Proximal Policy Optimization (PPO) - On-Policy

**Rede Neural:** Arquitetura **Actor-Critic** compartilhando o corpo da rede para extração de features, bifurcando em duas cabeças:
- **Critic Head:** Estima o Valor do estado (V-value)
- **Actor Head:** Define a política (ação)

**Controle Contínuo:** O Ator não escolhe uma ação discreta, mas sim a **Média (μ)** e o **Desvio Padrão (σ)** de uma **Distribuição Gaussiana** para os **4 motores das pernas** do robô simultaneamente.

### 🔹 Segurança Matemática

**PPO Clipping:** ε = 0.2 para evitar atualizações destrutivas de política

**Gradient Clipping:** `max_norm = 0.5` para conter explosões no Backpropagation

**Incerteza Mínima:** Adição de `1e-5` na camada de entropia para evitar colapsos matemáticos (Divisão por Zero / NaNs)

---

## 📊 A "Autópsia" do Treinamento (Post-Mortem)

Treinar o `BipedalWalker` com um PPO puro é um exercício extremo de **Hyperparameter Tuning**. O treinamento de **10.000 episódios** revelou fenômenos clássicos e fascinantes da literatura de Aprendizado por Reforço:

### 🔹 1. O Gênio Precoce (Episódios 1 a ~1.000)

O agente demonstrou uma capacidade de aprendizado incrivelmente rápida. A arquitetura matemática provou-se correta logo de início. O modelo saiu de pontuações de **-120** para picos isolados de **+222 pontos** em menos de 1.000 episódios, desenvolvendo uma marcha bípede funcional e quase cruzando a linha de chegada (+300 pts).

### 🔹 2. O Cérebro Derretido (Exploding Gradients)

Conforme o robô atingia a maestria de equilíbrio, sua "certeza" aumentava, fazendo o desvio padrão (σ) tender a zero. Ao tentar calcular o logaritmo da probabilidade em uma curva de sino infinitamente estreita, a matemática do PyTorch gerou valores infinitos, resultando em tensores corrompidos (`NaN`).

**Solução Implementada:** Injeção de ruído mínimo obrigatório e Gradient Clipping, que estancou a falha permanentemente.

### 🔹 3. O Esquecimento Catastrófico

Ao aplicar o conceito de **Warm Restart** (retornando a Taxa de Aprendizado para 100% após um longo ciclo de treino), o otimizador aplicou uma "volantada" muito brusca em uma rede que já possuía pesos refinados. A IA "desaprendeu" a caminhar em questão de 100 episódios, com a média despencando para **-100**.

### 🔹 4. A Armadilha da Recompensa (Reward Trap / Local Optimum)

Após o esquecimento catastrófico, o agente descobriu uma brecha matemática na função de recompensa da engine física:

- **Tentar andar e cair:** Penalidade de **-100 pontos**
- **Ficar imóvel no chão:** ~**-40 pontos** (gastando mínimo de energia)

**Conclusão do agente:** Como a Taxa de Aprendizado (LR Decay) decaiu e o agente perdeu a força de exploração necessária para se reerguer, a IA optou pelo **caminho de menor esforço e menor dor**. O modelo final estabilizou em uma estratégia de "preguiça", otimizando ativamente para tirar `-40` e evitar a dor absoluta da queda.

---

## 🚀 Como Executar

### 🔹 1. Instalação

Clone o repositório e instale as dependências:

```bash
pip install -r requirements.txt
```

**Aviso:** Requer a biblioteca `swig` e o pacote `box2d` do seu sistema operacional.

### 🔹 2. Treinamento

Para iniciar o treinamento do zero ou continuar a partir de um checkpoint existente:

```bash
python train.py
```

**Nota:** O script suporta Google Drive se rodado no Google Colab.

### 🔹 3. Avaliação (Visualização do Agente)

Para testar os pesos do modelo treinado e gerar um vídeo `.mp4` da corrida física da IA:

```bash
python evaluate.py
```

---

## 📌 Autor

**Álvaro Braz**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/alvaro-braz-cunha)

Projeto desenvolvido para fins de **estudo e portfólio profissional em Deep Reinforcement Learning e PPO**.
