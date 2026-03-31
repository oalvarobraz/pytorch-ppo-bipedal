import gymnasium as gym
import numpy as np

def main():
    env = gym.make("BipedalWalker-v3", render_mode="human")
    
    estado, info = env.reset()
    
    print("Diagnóstico do BipedalWalker:")
    print(f"Formato do Estado (Sensores): {env.observation_space.shape}")
    print(f"Limites da Ação (Motores): {env.action_space.low} até {env.action_space.high}")
    print("-" * 40)

    for step in range(100):
        # Gerando uma ação aleatória: 4 números entre -1 e 1
        acao = env.action_space.sample()
        
        proximo_estado, recompensa, finalizado, truncado, info = env.step(acao)
        
        if step % 20 == 0:
            print(f"Passo {step}:")
            print(f" -> Motores (Exemplo): {acao}")
            print(f" -> Velocidade Horizontal: {proximo_estado[2]:.2f}")
            print(f" -> Contato com o chão: {proximo_estado[8:10]} (Pé Esq/Dir)")
        
        if finalizado or truncado:
            break
            
    env.close()
    print("Teste concluído. O robô provavelmente caiu porque as ações eram aleatórias.")

if __name__ == "__main__":
    main()