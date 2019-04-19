import gym
import trading_env
import math
import matplotlib.pyplot as plt
import random
import numpy as np
import definitions
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected

def discount_rewards(rewards, discount_rate):
    discounted_rewards = np.zeros(len(rewards))
    cumulative_rewards = 0
    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * discount_rate
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]


amount_episodes = 10
amount_steps = 2000   # finalnie zeby przesunac wzdluz calego wykresu np jak wykres bedize miec 5000 danych - tu dac 5000.

liczba_danych = 200
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
######################--------------faza konstrukcyjna sieci neuronowej------------#####################################
########################################################################################################################

n_inputs = 3 # tu powinno byc tyle wejsc jak rozmiar tego bufora uczacego. wstepnie daje tylko 1 (close_price)
n_steps = 150 # liczba danych w ciagu w sieci rekurencyjnej
n_neurons = 200 # od czego uzależniam liczbe neuronów? chyba testowo...
n_hidden = 4
n_outputs = 3 # bo mam wyscie BUY, SELL HODL


learning_rate = 0.01

initializer = tf.contrib.layers.variance_scaling_initializer()

#X = tf.placeholder(tf.float32, [None, n_inputs])
X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
#y = tf.placeholder(tf.int32, [None])

basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons) # używam zwykłej komórki dla rnn
outputs, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

#hidden = fully_connected(X, n_hidden, activation_fn=tf.nn.elu, weights_initializer=initializer)
#logits = fully_connected(hidden, n_outputs, activation_fn=None)
#outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)

logits = tf.layers.dense(states, n_outputs)#, kernel_initializer=initializer)

out_softmax = tf.nn.softmax(logits)

buy_sell_hodl = tf.concat(axis=1, values=out_softmax)


#action = tf.multinomial([tf.log(buy_sell_hodl[0]), tf.log(buy_sell_hodl[1]), tf.log(buy_sell_hodl[2])], num_samples=1)
action = tf.multinomial(tf.log(buy_sell_hodl), num_samples=1)
#action = tf.squeeze(tf.multinomial(buy_sell_hodl, num_samples=1), axis=-1)
#y = tf.one_hot(action, depth=n_outputs)
#y = 1. - tf.to_float(action)
#y = action
y = tf.to_float(action)
#y = 1. - tf.to_float(action)# uzupelniam o prawdopodobienstwo docelowe - str 442. chodzi o to, ze docelowo musze miec
# prawdopodobienstwo o wartosci 1. Nie bardzo kumam o co tu chodzi.

#funkcja kosztu:
#xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits) # tu mam wyjscie sieci
xentropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits) # tu mam wyjscie sieci
#xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits) # tu mam wyjscie sieci

optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(xentropy)

gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []

for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32,shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients((grads_and_vars_feed))

init = tf.global_variables_initializer()

saver = tf.train.Saver()


########################################################################################################################
#######################--------------faza konstrukcyjna sieci neuronowej------------####################################
########################################################################################################################

# TODO: faza konstrukcyjna wstępnie zrobiona prawidłowo. Nie pokazuje żadnych błędów.

########################################################################################################################
#######################--------------faza wykonawcza sieci neuronowej---------------####################################
########################################################################################################################

a = 1

# TODO: faza wykonawcza wstępnie tylko przepisane, poprawic tak zeby dzialalo z tradingenv (zobacz od linii 220 poniżej

n_iterations = 250 # liczba przebiegów uczacych
n_max_steps = 100 # liczba kroków w episodzie
n_games_per_update = 10 # uczenie polityki co 10 kroków
save_iterations = 10 # co ile iteracji zapisuje model

discount_rate = 0.5

env = gym.make('trading_env-v0')


dane_do_analizy = dane_sinusoida


testowe_ceny_kupna = []
testowe_ceny_sprzedazy = []
tab_testowe_ceny_kupna = []
tab_testowe_ceny_sprzedazy = []


with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        print("\rIteration: {}".format(iteration), end="")
        print('\n')
        all_rewards = [] # wszystkie sekwencje pełnych nagród w każdym epizodzie
        all_gradients = [] # zachowane gradienty w każdym kroku każdego epizodu
        for game in range(n_games_per_update):
            current_rewards = [] # wszystkie pelne nagrodu z bieżącego epizodu
            current_gradients = [] # wszystkie gradienty z biezacego epizodu

            states = [] # przechowuje aktualny stan - czy kupione, sprzedane czy jak. Jest to jednym z wejsc na moją sieć.
            for i in range(liczba_danych):
                states.append(definitions.STATE_HAVE_USDT)

            # nizej rpzypadek gdy chce przejsc przez caly wykres uczacy, a nie tylko przez randomowe fragmenty.
            env.reset(kwota_usd_poczatkowa)
            n_max_steps = int(len(dane_do_analizy)/liczba_danych) # w przyblizeniu przechodze przez cale dane
            for step in range(n_max_steps):
                print("kolejny step = {0}".format(step))
                dane = [[float(dane_do_analizy[0 + step:liczba_danych + step][i]) + 1 for i in
                         range(len(dane_do_analizy[0 + step:liczba_danych + step]))],
                        [float(dane_do_analizy[0 + step:liczba_danych + step][i]) + 1 for i in
                         range(len(dane_do_analizy[0 + step:liczba_danych + step]))]]

                dane.append(states) #dodaje aktualny stan gdzie mam tylko usdt, zadnych btc.
                #plt.plot(dane[0])
               # plt.show()
            # niżej przypadek gdy nie chce przechodzic przez caly wykres a chce wyrywac z wykresu pewne randomowe miejsca do uczenia
            # env.reset(kwota_usd_poczatkowa)
            # for step in range(n_max_steps):
            #     print("kolejny step = {0}".format(step))
            #     rand_value = random.randint(0, len(dane_do_analizy))
            #     print('rand_value = {0}'.format(rand_value))
            #     dane = [[float(dane_do_analizy[0 + rand_value:liczba_danych + rand_value][i]) + 1 for i in range(len(dane_do_analizy[0 + rand_value:liczba_danych + rand_value]))], [float(dane_do_analizy[0 + rand_value:liczba_danych + rand_value][i]) + 1 for i in range(len(dane_do_analizy[0 + rand_value:liczba_danych + rand_value]))]]
            #
            #     # plt.plot(dane[0])
            #     # plt.show()


                """
                ########################################################################################################
                ########################################################################################################
                ################################---------------------------------------#################################
                ################################-----------------UWAGA-----------------#################################
                ################################---------------------------------------#################################
                ########################################################################################################
                ########################################################################################################
                
                Wyzej mam dane. dane to jakis ciag. np sinusoida. Do X_batch moze powinienem wpisywac jakis fragment tej 
                sinusoidy.
                kolejne X_batch to tablicza. w tej tablicy raczej powinienem miec powyzsze dane przesuniete o 1. 
                W ten sposob moge w calym x batch zrobic swego rodzaju symulacje dla calego przebiegu.
                
                
                
                Koniecznie zrobic jakies zabezpieczenie zeby nie sprzedawal jesli nie jest kupione i nie kupowal 
                gdy jest juz kupione. Moze po prostu dac to jako wejscie do sieci aktualny stan. i wtedy jesli sprobuje
                kupic gdy juz kupione, lub sprzedac gdy juz sprzedane - bardzo duzy ujemny reward
                
                
                """

                dane_2 = np.array(dane)
                dane_2 = dane_2.T
                X_batch = np.array(np.zeros((1, n_steps, n_inputs)))
                for i in range(len(dane_2) - n_steps):
                    row_data_train = dane_2[i:i + n_steps]
                    row_data_train = np.array(row_data_train)[None, :]

                    X_batch = np.concatenate([X_batch, row_data_train])

                X_batch = X_batch[1:]

                #plt.plot(X_batch[0])
                #plt.show()


                #action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                # tu nizej do X wpisac odpowiednio przeksztalcony wektor danych


                """
                
                UWAGA WAŻNE!!!!!!!!!!!!!!!!!!!!!!!
                
                JAK TU UZALEZNIC OBLICZANE ACTION VAL?
                W SENSIE... 
                jesli nie mam usdt to nie powinien mi kupowac. i gdy nie mam btc to nie powinien sprzedawac...
                
                Jak na razie zrobic tak jakby bez uwzgledniania tego - moze bedzie dzilac, (chociaz_watpie)
                
                O a moze jako jedno z wejsc zrobic aktualny stan ????????????
                
                tak robię w tej chwili!
                
                """


                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: X_batch})

                # action_val to jedna z 3 wartosci, 0, 1 lub 2 - dla kazdego X_batch[index] mam okreslona wartosc wyjscia 0 1 lub 2.
                # Teraz jak tego uzyc dalej?
                reward = []
                done = []


                """
                
                
                Tych action_val bedzie dokladnie tyle jaki rozmiar bedzie miec X_batch.
                a wiec musze przeleciec przez caly rozmiar x batch aby przejsc tak jakby przez caly wykres danych
                (patrz uwaga wyzej)
                
                Niżej zadbać, zeby w przypadku gdy nie mam btc - próbuje sprzedać - dać dużą karę.
                Gdy mam btc i próbuję kupić  dać dużą karę.
                
                """

                for i in range(len(action_val)-1):
                    #print("i = {0}, akcja = {1}".format(i, action_val[i][0]))
                    dane_close_price = []
                    for j in range(n_steps):
                        dane_close_price.append(list(X_batch[i][j])[0])
                    #print('czy ma teraz kupione czy sprzedane = ', list(X_batch[i][-1])[2])
                    reward_value, done_v, X_batch = env.step(action_val[i][0], list(X_batch[i][-1])[2], dane_close_price[-1], X_batch, i)
                    b = 1
                # for i in range(len(action_val)):
                #     print("action_val[{0}] = ".format(i), action_val[i][0])
                #     if action_val[i][0] == definitions.BUY and env.return_kwota_usdt() != 0:
                #         print("kwota usdt = ", env.return_kwota_usdt())
                #         cena_kupna = env.bought_btc(dane[0])
                #         print("cena_kupna = ", cena_kupna)
                #         reward_value, done_v = env.step(action_val[i][0])
                #         testowe_ceny_kupna.append(cena_kupna)
                #
                #     elif action_val[i] == definitions.SELL and env.return_kwota_btc() != 0:
                #         print("kwota btc = ", env.return_kwota_btc())
                #         cena_sprzedazy = env.sold_btc(dane[0])
                #         print("cena sprzedazy = ", cena_sprzedazy)
                #         reward_value, done_v = env.step(action_val[i][0])
                #         testowe_ceny_sprzedazy.append(cena_sprzedazy)
                #     elif action_val[i] == definitions.HODL:
                #         print("HOLDUJE")
                #         reward_value, done_v = env.step(action_val[i][0])


                    reward.append(reward_value)
                    done.append(done_v)
                    #print('i = ', i)
                    #print("reward = ", reward)
                    a = 1
                    if done[-1]:
                        break

                """
                
                Do current_rewards zapisuje wszystkie kolejne nagrody dla calego mojego przejscia przez wykres danych
                
                """

                current_rewards.append(reward)
                current_gradients.append(gradients_val)

            """

             Do all_rewards zapisuje wszystkie zbiory current rewards. czyli dla wszystkich przejsc przez wykres. 
             A wiec po prostu dla kazdego wylosowanego fragmentu danych w innym iteratorze tablicy.

             """

            all_rewards.append(current_rewards)
            all_gradients.append(current_gradients)
            a = 1
            print("len all_gradients = ", len(all_gradients))
            print("len all_rewards = ", len(all_rewards))
            print("len all_gradients[0] = ", len(all_gradients[0]))
            print("len all_rewards[0] = ", len(all_rewards[0]))
            print("len all_gradients[0][0] = ", len(all_gradients[0][0]))
            print("len all_rewards[0][0] = ", len(all_rewards[0][0]))
            print("len all_gradients[0][0][0] = ", len(all_gradients[0][0][0]))

        # na tym etapie polityka byla uruchomiona przez 10 epizodów
        # i jestesmy gotowi na jej zaktualizowanie za pomoca omawianego wczesniej algorytmu.


        # TU SKONCZYLEM POWIEDZMY

        # byc moze reward zle liczone jest - do sprawdzenia.
        #


        # UWAGA TU ZATRZYMAC LICZENIE I POROWNAC all_gradients z all_gradients z pliku  test_siec_samouczaca_.... (mam sssa ma pulpicue)

        for i in range(len(action_val)):
            all_rewards_ = discount_and_normalize_rewards(all_rewards[i], discount_rate=discount_rate)
            feed_dict = {}
            for var_index, gradient_placeholder in enumerate(gradient_placeholders):
                mean_gradients = np.mean([reward * all_gradients[game_index][step][var_index][i]
                                          for game_index, rewards in enumerate(all_rewards_)
                                          for step, reward in enumerate(rewards)], axis=0)
                feed_dict[gradient_placeholder] = mean_gradients
            sess.run(training_op, feed_dict=feed_dict)


        if iteration % save_iterations == 0:
            saver.save(sess, "./my_policy_net_pg.ckpt")

env.close()







# ########################################################################################################################
# #######################--------------faza wykonawcza sieci neuronowej---------------####################################
# ########################################################################################################################
#
# env = gym.make('trading_env-v0')
#
# totals = []
#
# testowe_ceny_kupna = []
# testowe_ceny_sprzedazy = []
# tab_testowe_ceny_kupna = []
# tab_testowe_ceny_sprzedazy = []
#
# dane_do_analizy = dane_sinusoida
# i=0
#
# for episode in range(amount_episodes):
#     episode_rewards = 0
#     env.reset(kwota_usd_poczatkowa)
#
#     for step in range(amount_steps):
#         dane = [float(dane_do_analizy[0+step:liczba_danych + step][i]) + 1 for i in range(len(dane_do_analizy[0 + step:liczba_danych + step]))]
#
#         action = przyklad_policy(dane)
#         # tu zamiast action = przyklad_policy(dane) powinna byc siec neuronowa
#
#         if action == definitions.BUY and env.return_kwota_usdt() != 0:
#             cena_kupna = env.bought_btc(dane)
#             testowe_ceny_kupna.append(cena_kupna)
#
#         elif action == definitions.SELL and env.return_kwota_btc() != 0:
#             cena_sprzedazy = env.sold_btc(dane)
#             testowe_ceny_sprzedazy.append(cena_sprzedazy)
#
#         else:
#             i += 1
#            # print("jałowo i = ", i)
#
#         reward, done = env.step(action)
#
#         episode_rewards += reward
#         if done:
#             break
#
#     print("episode iterator = ", episode)
#     totals.append(episode_rewards)
#     tab_testowe_ceny_kupna.append(testowe_ceny_kupna)
#     tab_testowe_ceny_sprzedazy.append(testowe_ceny_sprzedazy)
#     testowe_ceny_sprzedazy = []
#     testowe_ceny_kupna = []
#     i = 0
#
# print("\n\r wyniki koncowe:")
#
# print(np.mean(totals))
# print(np.std(totals))
# print(np.min(totals))
# print(np.max(totals))
#
#
#
# a = 1



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
