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
        self.kwota_btc = 100



    def step(self, action):
        a = 1
        zmiana_zysku = ((self.kwota_usdt - self.poczatkowa_wartosc_usdt) / self.poczatkowa_wartosc_usdt) * 100
        """
        jako argument pobiera to co moja siec bedzie robić (czyli buy, sell, hodl)
        zwracam chyba to, jak środowisko reaguje na daną akcję
        - może, jesli kupie to zeruje ilosc hajsu, zwiekszam ilosc btc, koryguje rewards;
        - jesli sprzedaje to zeruje ilsoc btc, zwiekszam ilosc hajsu, koryguje rewards;
        - jesli hodl, to koryguje rewards
        """

        self.step_iteration += 1

        if action == definitions.BUY:
            self.reward += 1

        elif action == definitions.HODL:
            a= 1

        elif action == definitions.SELL:
            if zmiana_zysku > 0:
                self.reward += zmiana_zysku
            elif zmiana_zysku < 0:
                self.reward -= zmiana_zysku
            else:
                self.reward += 1

        # w przypadku gdy zdarzy sie sytuacja ze strace 10% kapitalu - zatrzymuje dzialanie srodowiska
        # if zmiana_zysku < -4.98:
        #     self.done = True

        """
        zwracam observation, reward, done, info    
        """
        return self.reward, self.done

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

    def bought_btc(self, dane):
        self.kwota_btc = self.kwota_usdt / float(dane[0])

        print("kupiono", self.kwota_btc, "btc za", self.kwota_usdt)
        print("cena za btc = ", dane[0])
        self.kwota_usdt = 0

        return dane[0]

    def sold_btc(self, dane):
        self.kwota_usdt = self.kwota_btc * float(dane[0])
        print("sprzedano ", self.kwota_btc, "btc za", self.kwota_usdt)
        print("cena za btc = ", dane[0])
        self.kwota_btc = 0

        return dane[0]

    def return_kwota_usdt(self):
        return self.kwota_usdt

    def return_kwota_btc(self):
        return self.kwota_btc

