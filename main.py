import gym
import trading_env
import math
import matplotlib.pyplot as plt
import random
import numpy as np
import definitions
import tensorflow as tf



amount_episodes = 10
amount_steps = 2000   # finalnie zeby przesunac wzdluz calego wykresu np jak wykres bedize miec 5000 danych - tu dac 5000.

liczba_danych = 2000
kwota_usd_poczatkowa = 1000



open_price = []
high_price = []
low_price = []
close_price = []
volume = []
date_unix = []

date_unix_index = 0
date_index = 1
open_price_index = 3
high_price_index = 4
low_price_index = 5
close_price_index = 6
volume_index = 7
i = 0

plik = open('dane_uczace.csv', 'r')
dane_pobrane_z_pliku = plik.read()
dane_bez_pierwszych_linii = dane_pobrane_z_pliku[dane_pobrane_z_pliku.index("Volume") + len('Volume') + 1:]
dane_rozdzielone = dane_bez_pierwszych_linii.split(',')

#self.date_unix.append(dane_rozdzielone[0])
while volume_index < len(dane_rozdzielone)-1:  # i <= len(dane_rozdzielone):#-1:# -1 to ze do tylu ile trzeba bez podawania rozmiaru
    open_price.append((dane_rozdzielone[open_price_index]))
    high_price.append((dane_rozdzielone[high_price_index]))
    low_price.append((dane_rozdzielone[low_price_index]))
    close_price.append((dane_rozdzielone[close_price_index]))

    # tu w volume_zakloconym mam ostatnia wartosc i pierwsza w nastepnej linii. te operacje pozwalaja mi rozdzielic to
    volume_zaklocone = (dane_rozdzielone[volume_index])
    volume_1 = volume_zaklocone.split('\n')
    volume.append(volume_1[0])
    date_unix.append(str(int(volume_1[1]) + 60)) #odejmuje 60 bo tu volume_1[1] to data_unix ale z nastepnego wiersza. a chce miec wartosc dla pierwszego wiersza wiec o 60s wczesniej.
    date_unix_index += 7
    date_index += 7
    open_price_index += 7
    high_price_index += 7
    low_price_index += 7
    close_price_index += 7
    volume_index += 7
    i += 1


dane_sinusoida = [math.sin(0.01 * x) for x in range(10000)]
print("first", close_price[0])
print("last = ",close_price[-1])

# TODO: jak juz zrobie sieć neuronową dla danych sinusoidalnych - poniższe wykomentować i zacząć testy na danych open, high low, close, volume.
# dane_sinusoida = close_price[::-1]

a = 1
def przyklad_policy(dane_do_analizy):
    if dane_do_analizy[0] < 0.01:
        return 1
    elif dane_do_analizy[0] > 1.99:
        return -1
    else:
        return 0

########################################################################################################################
########################################################################################################################
##########################################faza konstrukcyjna sieci neuronowej###########################################
########################################################################################################################
########################################################################################################################


n_inputs = 4
n_steps = 300 # liczba danych w ciagu w sieci rekurencyjnej
n_neurons = 200
n_outputs = 3 # bo mam wyscie BUY, SELL HODL

learning_rate = 0.01

X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placegolder(tf.float32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) # używam zwykłej komórki dla rnn
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

logits = tf.layers.dense(states, n_outputs)
xentropy - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)

# TODO:  dokończyć

########################################################################################################################
########################################################################################################################
##########################################faza konstrukcyjna sieci neuronowej###########################################
########################################################################################################################
########################################################################################################################


env = gym.make('trading_env-v0')

totals = []

testowe_ceny_kupna = []
testowe_ceny_sprzedazy = []
tab_testowe_ceny_kupna = []
tab_testowe_ceny_sprzedazy = []

dane_do_analizy = dane_sinusoida
i=0

for episode in range(amount_episodes):
    episode_rewards = 0
    env.reset(kwota_usd_poczatkowa)

    for step in range(amount_steps):
        dane = [float(dane_do_analizy[0+step:liczba_danych + step][i]) + 1 for i in range(len(dane_do_analizy[0 + step:liczba_danych + step]))]

        action = przyklad_policy(dane)

        if action == definitions.BUY and env.return_kwota_usdt() != 0:
            cena_kupna = env.bought_btc(dane)
            testowe_ceny_kupna.append(cena_kupna)

        elif action == definitions.SELL and env.return_kwota_btc() != 0:
            cena_sprzedazy = env.sold_btc(dane)
            testowe_ceny_sprzedazy.append(cena_sprzedazy)

        else:
            i += 1
           # print("jałowo i = ", i)

        reward, done = env.step(action)

        episode_rewards += reward
        if done:
            break

    print("episode iterator = ", episode)
    totals.append(episode_rewards)
    tab_testowe_ceny_kupna.append(testowe_ceny_kupna)
    tab_testowe_ceny_sprzedazy.append(testowe_ceny_sprzedazy)
    testowe_ceny_sprzedazy = []
    testowe_ceny_kupna = []
    i = 0

print("\n\r wyniki koncowe:")

print(np.mean(totals))
print(np.std(totals))
print(np.min(totals))
print(np.max(totals))



a = 1



#
#
# amount_episodes = 500
# amount_steps = 500
#
# totals = []
#
# for episode in range(amount_episodes):
#     episode_rewards = 0
#     obs = env.reset()
#     for step in range(amount_steps):
#         action = policy(obs)
#         obs, reward, done, info = env.step(action)
#         episode_rewards += reward
#         if done:
#             break
#     totals.append(episode_rewards)
