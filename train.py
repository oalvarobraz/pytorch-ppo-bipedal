import os
import torch
import gymnasium as gym
import numpy as np

from src.model import ActorCritic
from src.agent import PPOAgent

def main():
    print("Iniciando o Treinamento do BipedalWalker (PPO)...")

    caminho_drive = "/content/drive/MyDrive/Bipedal_PPO"
    os.makedirs(caminho_drive, exist_ok=True)
    caminho_salvamento = os.path.join(caminho_drive, "ppo_bipedal.pth")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Aceleração CUDA ativada!")
    else:
        device = torch.device("cpu")
        print("Rodando na CPU.")

    env = gym.make("BipedalWalker-v3")
    num_sensores = env.observation_space.shape[0]
    num_motores = env.action_space.shape[0]

    modelo = ActorCritic(num_inputs=num_sensores, num_actions=num_motores)
    
    if os.path.exists(caminho_salvamento):
        modelo.load_state_dict(torch.load(caminho_salvamento))
        print("Pesos recuperados do Drive! Continuando o treinamento...")

    agente = PPOAgent(
        modelo=modelo,
        lr_ator=3e-4,
        lr_critico=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        epocas=10,
        device=device
    )

    EPISODIOS = 10000
    historico_pontuacoes = []
    melhor_pontuacao_isolada = 100.0
    caminho_campeao = os.path.join(caminho_drive, "ppo_bipedal_campeao.pth")
    
    for episodio in range(1, EPISODIOS + 1):
        estado, info = env.reset()
        done = False
        pontuacao_episodio = 0

        while not done:
            acao, log_prob, valor = agente.agir(estado)
            proximo_estado, recompensa, finalizado, truncado, info = env.step(acao)
            done = finalizado or truncado

            velocidade_x = proximo_estado[2]
            
            if velocidade_x < 0.1:
                recompensa -= 0.5
            
            agente.lembrar(estado, acao, log_prob, recompensa, done, valor)
            pontuacao_episodio += recompensa
            estado = proximo_estado

        agente.treinar()

        agente.decair_lr(episodio, EPISODIOS)

        historico_pontuacoes.append(pontuacao_episodio)
        media_100 = np.mean(historico_pontuacoes[-100:]) if len(historico_pontuacoes) >= 100 else np.mean(historico_pontuacoes)

        if episodio % 10 == 0:
            print(f"Episódio: {episodio}/{EPISODIOS} | Pts: {pontuacao_episodio:.1f} | Média(100): {media_100:.1f}")

        if pontuacao_episodio > melhor_pontuacao_isolada:
            melhor_pontuacao_isolada = pontuacao_episodio
            torch.save(agente.modelo.state_dict(), caminho_campeao)
            print(f"NOVO RECORDE DA CORRIDA: {melhor_pontuacao_isolada:.1f} pontos! Cérebro salvo no Drive!")
            
        if episodio % 100 == 0:
            torch.save(agente.modelo.state_dict(), caminho_salvamento)
            print(f"Backup salvo no Drive: {caminho_salvamento}")
        
        if media_100 >= 300:
            print(f"O robô atingiu a maestria no episódio {episodio}!")
            torch.save(agente.modelo.state_dict(), caminho_salvamento)
            break

    env.close()

if __name__ == "__main__":
    main()