import torch
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from torch.utils.tensorboard import SummaryWriter

from agent import *
from dqn import *
from replaybuffer import *

timestamp = datetime.datetime.now().strftime('%Y-%b-%d %H:%M')

writer = SummaryWriter(f"runs/{timestamp}")

file = "market_15mn.csv"

ACTIONS = {
    "buy":0,
    "sell":1,
    "hold":2
}

agent_config = {
    "delta":0.00001,
    "epsilon_clip":0.4
    
}

buffer_config = {
    "Buffer_size":10000,
    "Batch_size": 1000,
    "Buffer_epsilon":0.001,
    "Buffer_alpha":0.2
}
network_config = {
    
}


class Wallet:
    def __init__(self, amount):
        self.USD_wallet = amount
        self.BTC_wallet = 0
        self.BTCUSD_rate = None
        self.USDBTC_rate = None
        
    def to_USD(self, amount):
        return amount*self.BTCUSD_rate
        
    def to_BTC(self, amount):
        return amount*self.USDBTC_rate
        
    def set_rate(self, rate):
        self.BTCUSD_rate = rate
        self.USDBTC_rate = 1/rate
        
        
class Analyser:
    def __init__(self):
        
        self.market_analysis_network = Network(input_dimension=5, output_dimension=1) # some GRU, LSTM network
        
    def preprocess(data):
        
        processed_data = (stock_price, RSI, shares, wallet)
        
        return processed_data
    
    def analyse(self, data):
        
#         state = (stock_price, RSI, shares, wallet)
        # add LSTM embedding on latent space to make end to end
        
#         self.market_analysis_network(data)

        market_price, rsi = data
        
        MA = (pd.DataFrame(market_price).rolling(10).sum()/10)
        
        MA = MA.interpolate(method='linear', limit_direction='both')
        
        trend = (MA["Close"].iloc[-1] - MA["Close"].iloc[0])/len(MA)
        
        return market_price.iloc[-1], rsi.iloc[1], trend
    
    def train_analyser(self, data):
        
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()
    
class Trader: # Performs analysis and sends trades
    def __init__(self, amount, agent_config, buffer_config):
        self.wallet = Wallet(amount)
        
        self.analyser = Analyser()
        
        self.agent = Agent(agent_config, buffer_config)
        
        self.state = None
        
        self.action = None
        
        self.market_idx = 0
        
        self.market = pd.read_csv(file)
        
        
    def get_data(self):
        return data
    
    def get_market(self, pos, nb = 0):
        Close_price, rsi = self.market["Close"][pos-nb:pos], self.market["rsi"][pos-nb:pos]
        return Close_price, rsi
    

    def make_trade(self, state, action):
        # State : stock price, RSI, (number of shares, share price), reward
        # reward is money made from action --> new_wallet - past wallet
        
#         stock_price, RSI, shares, reward = state

        fee = 0.75
        
        if action == ACTIONS["buy"]:
            if state[4] - fraction * state[0] >0:
                self.wallet.BTC_wallet += fraction
                self.wallet.USD_wallet -= fraction * state[0] - fee * fraction
            else:
                pass


        elif action == ACTIONS["sell"]:
            if state[3] - fraction >=0:
                self.wallet.BTC_wallet -= fraction
                self.wallet.USD_wallet += fraction * state[0] - fee * fraction
            else:
                self.wallet.BTC_wallet -= state[3]
                self.wallet.USD_wallet += state[3] * state[0] - fee * state[3]
                state[3] = 0

        elif action == ACTIONS["hold"]:
            pass
            
        else:
            print(action)
                

        # Wait for next price, rsi
        next_price, next_rsi, next_trend = self.analyser.analyse(self.get_market(self.market_idx+1, nb=30))
        
        next_state = (next_price, next_rsi, next_trend, self.wallet.BTC_wallet, self.wallet.USD_wallet)

        reward = (next_state[4] - state[4]) + (next_price * (next_state[3]-state[3]))
        
#         reward = (self.wallet.USD_wallet - 100 ) + self.wallet.BTC_wallet*state[0]
        
        return next_state, reward
        
    def decision_making(self):
        
        data = self.get_market(self.market_idx, nb=30)
        
        if len(data[0]) < 30:
            self.market_idx +=1
            return None
        
#         data = self.analyser.preprocess(data)
        
        market = self.analyser.analyse(data)
        
        self.wallet.set_rate(market[0])
        
        self.state = [market[0], market[1], market[2], self.wallet.BTC_wallet, self.wallet.USD_wallet]
        
        action = self.agent.get_next_action(self.state)
        
#         print(self.state, action)
        
        next_state, reward = self.make_trade(self.state, action)
        
        self.agent.set_next_state(next_state, reward)
        self.market_idx +=1
        
        return reward
        
if __name__ == "__main__":    
    trader = Trader(100, agent_config, buffer_config)
    
    wallet_ = []
    profit = []
    epsilon = []
    running = True

    fraction = 10 * trader.market["Close"][0]
    # p = 0.1

    while running:
        try:

            reward = trader.decision_making()
            if reward == None:
                continue

            wallet_.append(trader.state[4])

            trader.wallet.set_rate(trader.state[0])

    #         USD = p * (trader.wallet.USD_wallet) + (trader.wallet.to_USD(trader.wallet.BTC_wallet))

    #         fraction = trader.wallet.to_BTC(USD)


            profit.append((trader.wallet.USD_wallet - 100) + (trader.wallet.to_USD(trader.wallet.BTC_wallet)))

            epsilon.append(trader.agent.epsilon)

            writer.add_scalar('profit', (trader.wallet.USD_wallet - 100) + (trader.wallet.to_USD(trader.wallet.BTC_wallet)), trader.market_idx)
            writer.add_scalar('USD wallet',
                                trader.wallet.USD_wallet,
                                trader.market_idx)
            
            writer.add_scalar('BTC wallet',
                                trader.wallet.BTC_wallet,
                                trader.market_idx)

            writer.add_scalar('Allowed investment',
                                trader.wallet.to_USD(fraction),
                                trader.market_idx)

            writer.add_scalar('epsilon',
                                trader.agent.epsilon,
                                trader.market_idx)
            writer.add_scalar('BTCUSDT',
                                trader.market["Close"][trader.market_idx],
                                trader.market_idx)
            writer.add_scalar('reward',
                                reward,
                                trader.market_idx)

            if trader.market_idx%1000 == 0:
                print(f"Step: {trader.market_idx}/{len(trader.market)}\nProfit: {(trader.wallet.USD_wallet - 100) + (trader.wallet.to_USD(trader.wallet.BTC_wallet))}\nState: {trader.state}\n")


            fraction = trader.wallet.to_BTC(10)
    #         if trader.market_idx%1000 == 0:
    #             p += 0.01


        except:
            running=False

    fig,(ax1, ax2, ax3) = plt.subplots(1,3, figsize=(15,5))

    n=100

    ax1.plot(profit)
    (pd.DataFrame(profit).rolling(n).sum()/n).plot(ax=ax1)
    ax1.set_title("profit")
    ax1.plot(epsilon)

    ax2.plot(wallet_)
    (pd.DataFrame(wallet_).rolling(n).sum()/n).plot(ax=ax2)

    ax2.set_title("wallet")
    
    fig.savefig(f"plot_{timestamp}.pdf")
