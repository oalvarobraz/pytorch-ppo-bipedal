import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic, self).__init__()
        
        # ATOR (O que decide o torque)
        self.actor_fc1 = nn.Linear(num_inputs, hidden_size)
        self.actor_fc2 = nn.Linear(hidden_size, hidden_size)
        self.mu_head = nn.Linear(hidden_size, num_actions)      # Média do torque
        self.sigma_head = nn.Linear(hidden_size, num_actions)   # Desvio padrão (incerteza)

        self.softplus = nn.Softplus()

        # CRÍTICO (O que avalia a situação)
        self.critic_fc1 = nn.Linear(num_inputs, hidden_size)
        self.critic_fc2 = nn.Linear(hidden_size, hidden_size)
        self.value_head = nn.Linear(hidden_size, 1)             # Nota do estado

    def forward(self, x):
        # Processamento do Ator
        a = F.relu(self.actor_fc1(x))
        a = F.relu(self.actor_fc2(a))
        
        # A média (mu) passa por Tanh para garantir que o torque esteja entre -1 e 1
        mu = torch.tanh(self.mu_head(a))
        
        # O desvio padrão (sigma) deve ser sempre positivo.
        # Usamos Softplus para transformar qualquer número em um valor > 0.
        sigma = self.softplus(self.sigma_head(a))
        
        # Processamento do Crítico-
        v = F.relu(self.critic_fc1(x))
        v = F.relu(self.critic_fc2(v))
        value = self.value_head(v)
        
        return mu, sigma, value

    def selecionar_acao(self, estado):
        """
        Gera uma distribuição Gaussiana baseada no estado atual 
        e sorteia uma ação (torque) dela.
        """
        mu, sigma, value = self.forward(estado)
        
        # Criamos a distribuição Normal (Gaussiana)
        dist = Normal(mu, sigma)
        
        # Sorteamos a ação (amostragem)
        acao = dist.sample()
        
        # Calculamos o logaritmo da probabilidade
        log_prob = dist.log_prob(acao).sum(dim=-1, keepdim=True)
        
        return acao, log_prob, value