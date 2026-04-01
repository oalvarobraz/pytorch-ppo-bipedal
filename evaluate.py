import os
import torch
import gymnasium as gym
import imageio
from src.model import ActorCritic

def main():
    print("Preparando a câmera para filmar o robô...")
    
    os.makedirs("assets", exist_ok=True)
    caminho_video = "assets/bipedal_agent.mp4"
    caminho_pesos = "checkpoints/ppo_bipedal.pth"

    env = gym.make("BipedalWalker-v3", render_mode="rgb_array")
    num_sensores = env.observation_space.shape[0]
    num_motores = env.action_space.shape[0]

    modelo = ActorCritic(num_inputs=num_sensores, num_actions=num_motores)

    try:
        modelo.load_state_dict(torch.load(caminho_pesos, map_location=torch.device('cpu')))
        modelo.eval()
        print("Cérebro treinado carregado com sucesso!")
    except FileNotFoundError:
        print(f"Erro: Arquivo '{caminho_pesos}' não encontrado. Você precisa treinar primeiro!")
        return

    estado, info = env.reset()
    done = False
    pontuacao = 0
    frames = []

    print("🎬 Gravando a corrida...")

    while not done:
        frame = env.render()
        frames.append(frame)

        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            mu, sigma, valor = modelo(estado_tensor)
            acao = mu.numpy()[0] 
        
        estado, recompensa, finalizado, truncado, info = env.step(acao)
        done = finalizado or truncado
        pontuacao += recompensa

    env.close()

    print(f"Fim da corrida! Pontuação Total: {pontuacao:.1f}")
    print(f"Costurando {len(frames)} frames. Isso pode levar alguns segundos...")
    
    imageio.mimsave(caminho_video, frames, fps=30, macro_block_size=None)
    print(f"Sucesso! Vídeo salvo em: {caminho_video}.")

if __name__ == "__main__":
    main()