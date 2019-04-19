import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import matplotlib.pyplot as plt
import numpy as np

import definitions

"""

Na ogół action obliczane bedzie przez siec neuronowa (sell, buy, hodl)
action jest argumentem funkcji step
funkcja step oblicza nagrode, cos jeszcze?


Co zakładam:
 - Actions - BYY, SELL, HODL
wyłącznie kupuje sprzedaje lub hodluję

 - Reward
Gdy hodluje to nie wiekszam rewards
gdy kupuje i sprzedaje z zyskiem - reward dodatni
gdy kupuje i sprzedaje ze strata - reward ujemny

 - observation (self.obs)
tyo dane wejsciowe do srodowiska

"""

class TradingEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self):
        a = 1
        self.reward = 0
        self.step_iteration = 0
        self.obs = 0
        self.done = 0
        self.info = 0
        self.liczba_danych = 100
        self.poczatkowa_wartosc_usdt = 0

        self.kwota_usdt = 1000
        self.kwota_btc = 0



    def step(self, action, state, dane_close_price, X_batch, iteracja):
        a = 1

        """
        jako argument pobiera to co moja siec bedzie robić (czyli buy, sell, hodl)
        zwracam chyba to, jak środowisko reaguje na daną akcję
        - może, jesli kupie to zeruje ilosc hajsu, zwiekszam ilosc btc, koryguje rewards;
        - jesli sprzedaje to zeruje ilsoc btc, zwiekszam ilosc usdt, koryguje rewards;
        - jesli hodl, to koryguje rewards
        """

        self.step_iteration += 1
        print("self.iteration = ", self.step_iteration)

        if action == definitions.BUY and state == definitions.STATE_HAVE_USDT:
            print('BUY')
            self.reward += 1
            self.bought_btc(dane_close_price)

            # modyfikuje X_Batch
            # tu nastepne X_batch stan dac jako ze definitions.STATE_HAVE_BTC
            for i in range(len(X_batch)):
                X_batch[iteracja+1][i][2] = definitions.STATE_HAVE_BTC
            # modyfikuje X_Batch

        elif action == definitions.SELL and state == definitions.STATE_HAVE_BTC:
            print('SELL')
            # modyfikuje X_Batch
            # tu nastepne X_batch stan dac jako ze definitions.STATE_HAVE_USDT
            for i in range(len(X_batch)):
                X_batch[iteracja+1][i][2] = definitions.STATE_HAVE_USDT
            # modyfikuje X_Batch

            self.sold_btc(dane_close_price)

            zmiana_zysku = ((self.kwota_usdt - self.poczatkowa_wartosc_usdt) / self.poczatkowa_wartosc_usdt) * 100

            if zmiana_zysku > 0:
                self.reward += zmiana_zysku
                print('\n\rzmiana zysku =',zmiana_zysku, '\n\r')
            elif zmiana_zysku < 0:
                self.reward -= zmiana_zysku
                print('\n\rzmiana zysku =', zmiana_zysku, '\n\r')
            else:
                self.reward += 1

            # w przypadku gdy zdarzy sie sytuacja ze strace 10% kapitalu - zatrzymuje dzialanie srodowiska
            # if zmiana_zysku < -4.98:
            #     self.done = True



        elif action == definitions.HODL:
            self.reward += 1
            print('HODL')
            # modyfikuje X_Batch
            # tu nastepne X_batch stan dac taki jak biezacy czyli jak state
            for i in range(len(X_batch)):
                X_batch[iteracja+1][i][2] = state
            # modyfikuje X_Batch


        # jesli nie spelniam wyzej wymienionych warunkow to duza kara.
        # Tzn PO PROSTU  jesli nie spelnia warunkow zwiazanych ze sprzedaza lub kupnem powinien hodlowac.
        else:
            print('TU NIE POWINIEN WCHODZIC JAK JUZ SIE W MIARE NAUCZY')
            # modyfikuje X_Batch
            # tu nastepne X_batch stan dac taki jak biezacy czyli jak state
            for i in range(len(X_batch)):
                X_batch[iteracja+1][i][2] = state
            # modyfikuje X_Batch

            self.reward -= 500

        """
        zwracam observation, reward, done, info    
        """
        return self.reward, self.done, X_batch

    def reset(self, poczatkowa_wartosc_usdt):
        a = 1
        self.kwota_usdt = poczatkowa_wartosc_usdt
        self.poczatkowa_wartosc_usdt = poczatkowa_wartosc_usdt
        self.kwota_btc = 0

        """
        Zeruję sobie reward i powinienem jakos pobrac dane inicjujace. konkretniej open, hi lo close itd. 
        Tu wykorzystac te jak budowalem ten bufor danych. podczas resetu do obs wpisac pierwsze np 300 danych.

        """

       # self.liczba_danych = liczba_danych
        self.done = False
        self.reward = 0
        self.step_iteration = 0
       # self.dane_do_analizy = [dane[0:self.liczba_danych][i]+1 for i in range(len(dane[0:self.liczba_danych]))]
       # print(np.min(self.dane_do_analizy))
       # return self.dane_do_analizy

    def render(self, mode='human', close=False):
        a = 1

    def bought_btc(self, dane_close_price):
        self.kwota_btc = self.kwota_usdt / float(dane_close_price)

        print("kupiono", self.kwota_btc, "btc za", self.kwota_usdt)
        print("cena za btc = ", dane_close_price)
        self.kwota_usdt = 0

        return dane_close_price

    def sold_btc(self, dane_close_price):
        self.kwota_usdt = self.kwota_btc * float(dane_close_price)
        print("sprzedano ", self.kwota_btc, "btc za", self.kwota_usdt)
        print("cena za btc = ", dane_close_price)
        self.kwota_btc = 0

        return dane_close_price

    def return_kwota_usdt(self):
        return self.kwota_usdt

    def return_kwota_btc(self):
        return self.kwota_btc

