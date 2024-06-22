import gymnasium

def Env(**config):
    return gymnasium.make("CartPole-v1")