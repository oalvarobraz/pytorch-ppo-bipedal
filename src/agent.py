import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from torch.distributions import Normal

class PPOAgent:
    def __init__(self, modelo, lr_ator=3e-4, lr_critico=1e-3, gamma=0.99, clip_epsilon=0.2, epocas=10, device='cpu'):
        self.modelo = modelo.to(device)
        self.device = device
        
        self.otimizador = optim.Adam([
            {'params': [p for n, p in modelo.named_parameters() if 'actor' in n or 'mu' in n or 'sigma' in n], 'lr': lr_ator},
            {'params': [p for n, p in modelo.named_parameters() if 'critic' in n or 'value' in n], 'lr': lr_critico}
        ])

        self.lr_ator_inicial = lr_ator
        self.lr_critico_inicial = lr_critico
        
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epocas = epocas
        
        self.memoria = []

    def agir(self, estado):
        estado_tensor = torch.tensor(estado, dtype=torch.float32).unsqueeze(0).to(self.device)
        
        # O Ator sorteia a ação, e o Crítico dá a nota para o estado atual
        with torch.no_grad():
            acao, log_prob, valor = self.modelo.selecionar_acao(estado_tensor)
            
        return acao.cpu().numpy()[0], log_prob.cpu().item(), valor.cpu().item()

    def lembrar(self, estado, acao, log_prob, recompensa, done, valor):
        self.memoria.append((estado, acao, log_prob, recompensa, done, valor))

    def limpar_memoria(self):
        self.memoria = []

    def decair_lr(self, passo_atual, max_passos):
        fracao = 1.0 - (passo_atual - 1.0) / max_passos
        
        fracao = max(fracao, 0.1)
        
        self.otimizador.param_groups[0]['lr'] = self.lr_ator_inicial * fracao
        self.otimizador.param_groups[1]['lr'] = self.lr_critico_inicial * fracao

    def treinar(self):
        if len(self.memoria) == 0:
            return

        # Desempacota a safra de memórias
        estados = torch.tensor(np.array([m[0] for m in self.memoria]), dtype=torch.float32).to(self.device)
        acoes = torch.tensor(np.array([m[1] for m in self.memoria]), dtype=torch.float32).to(self.device)
        log_probs_antigos = torch.tensor([m[2] for m in self.memoria], dtype=torch.float32).to(self.device)
        recompensas = [m[3] for m in self.memoria]
        dones = [m[4] for m in self.memoria]
        valores_antigos = torch.tensor([m[5] for m in self.memoria], dtype=torch.float32).to(self.device)

        # Calcula as Recompensas Descontadas (Retorno Real)
        retornos = []
        retorno_acumulado = 0
        for recompensa, done in zip(reversed(recompensas), reversed(dones)):
            if done:
                retorno_acumulado = 0
            retorno_acumulado = recompensa + (self.gamma * retorno_acumulado)
            retornos.insert(0, retorno_acumulado)
        
        retornos = torch.tensor(retornos, dtype=torch.float32).to(self.device)
        
        # Calcula a Vantagem (Advantage) = O que aconteceu DE FATO menos o que o Crítico ESPERAVA
        vantagens = retornos - valores_antigos
        # Normalização da Vantagem (Ajuda a estabilizar o treino)
        vantagens = (vantagens - vantagens.mean()) / (vantagens.std() + 1e-8)

        # Treina a rede por várias Épocas usando a MESMA safra
        for _ in range(self.epocas):
            mu, sigma, valores_novos = self.modelo(estados)
            dist = Normal(mu, sigma)
            
            log_probs_novos = dist.log_prob(acoes).sum(dim=-1)
            
            razao = torch.exp(log_probs_novos - log_probs_antigos)

            surrogate1 = razao * vantagens
            
            surrogate2 = torch.clamp(razao, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * vantagens
            
            # Perda do Ator (Queremos MAXIMIZAR o surrogate, então minimizamos o negativo)
            perda_ator = -torch.min(surrogate1, surrogate2).mean()
            
            # Perda do Crítico (Erro Quadrático Médio entre a previsão e o retorno real)
            perda_critico = nn.MSELoss()(valores_novos.squeeze(), retornos)
            
            entropia = dist.entropy().mean()
            
            perda_total = perda_ator + (0.5 * perda_critico) - (0.02 * entropia)
            
            self.otimizador.zero_grad()
            perda_total.backward()
            nn.utils.clip_grad_norm_(self.modelo.parameters(), max_norm=0.5)
            self.otimizador.step()

        self.limpar_memoria()
