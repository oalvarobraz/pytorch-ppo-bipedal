import os
import torch
import gymnasium as gym
import numpy as np

from src.model import ActorCritic
from src.agent import PPOAgent

def main():
    print("Iniciando o Treinamento do BipedalWalker (PPO)...")

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Aceleração CUDA ativada!")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Aceleração MPS (Apple Silicon) ativada!")
    else:
        device = torch.device("cpu")
        print("Rodando na CPU.")

    env = gym.make("BipedalWalker-v3")
    num_sensores = env.observation_space.shape[0]
    num_motores = env.action_space.shape[0]

    modelo = ActorCritic(num_inputs=num_sensores, num_actions=num_motores)
    agente = PPOAgent(
        modelo=modelo,
        lr_ator=3e-4,
        lr_critico=1e-3,
        gamma=0.99,
        clip_epsilon=0.2,
        epocas=10,
        device=device
    )

    EPISODIOS = 5000
    historico_pontuacoes = []
    
    os.makedirs('checkpoints', exist_ok=True)
    caminho_salvamento = "checkpoints/ppo_bipedal.pth"

    for episodio in range(1, EPISODIOS + 1):
        estado, info = env.reset()
        done = False
        pontuacao_episodio = 0

        while not done:
            acao, log_prob, valor = agente.agir(estado)
            
            proximo_estado, recompensa, finalizado, truncado, info = env.step(acao)
            done = finalizado or truncado
            
            agente.lembrar(estado, acao, log_prob, recompensa, done, valor)
            
            pontuacao_episodio += recompensa
            estado = proximo_estado

        agente.treinar()

        historico_pontuacoes.append(pontuacao_episodio)
        media_100 = np.mean(historico_pontuacoes[-100:]) if len(historico_pontuacoes) >= 100 else np.mean(historico_pontuacoes)

        print(f"Episódio: {episodio}/{EPISODIOS} | Pontuação: {pontuacao_episodio:.1f} | Média(100): {media_100:.1f}")

        if episodio % 500 == 0:
            torch.save(agente.modelo.state_dict(), caminho_salvamento)
            print(f"Checkpoint salvo no episódio {episodio}!")
        
        if media_100 >= 300:
            print(f"Vitória! O robô aprendeu a andar como um humano no episódio {episodio}!")
            torch.save(agente.modelo.state_dict(), caminho_salvamento)
            break

    env.close()

if __name__ == "__main__":
    main()