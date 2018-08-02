r"""Implementace klasifikacni konvolucni neuronove site, uceni CNN."""
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import time

"""Iniciace excel souboru pro ukládání dat"""
list_categories = ['1 spine', '2 heart', '3 left lung', '4 head', '5 liver', '6 aorta',
                   '7 left kidney', '8 spleen', '9 right lung', '10 rigth kidney', '11 background']

"""Definice parametru vrstev"""
# 1. konvolucni vrstva
size_filter1 = 9          # filtry velikosti 5 x 5 px
num_filter1 = 16          # pocet filtru ve vrstve 16
# 2. konvolucni vrstva
size_filter2 = 9             # filtry velikosti 5 x 5 px
num_filter2 = 32             # pocet filtru ve vrstve 32
# 3. konvolucni vrstva
size_filter3 = 11             # filtry velikosti 5 x 5 px
num_filter3 = 64             # pocet filtru ve vrstve 64
# 4. konvolucni vrstva
size_filter4 = 11             # filtry velikosti 5 x 5 px
num_filter4 = 128            # pocet filtru ve vrstve 64
# plne propojena vrstva
size_fully1 = 128            # pocet neuronu 1. plne propojene vrstvy 128


def _parse_function(example_proto):
    """Funkce pro zisk dat z tfrecordu.

    - Pouziti u fce dataset.map(_parse_function)
    """
    features = {"image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.cast(image, tf.float32) * (1. / 255)  # prevod na float [0.0, 1.0]
    image.set_shape([128 * 128])
    label = tf.cast(parsed_features['label'], tf.int64)
    one_hot = tf.one_hot(label - 1, num_class, dtype=tf.float32)
    return image, one_hot


"""Vstupni data"""

with tf.name_scope("Trenovaci_data"):
    # Tvorba datasetu, ktery nacte examples z TFRecord a vezme roi a znaceni pro CNN
    num_class = 11  # pocet trid pro klasifikaci
    batch_size = 4  # pocet obr z 1 tridy pro 1 beh trenovani site
    # delka datasetu pro jednotlive skupiny (kazda ma jiny pocet oblasti)
    buffer_size_list = [10000, 10000, 10000, 600, 10000, 10000, 10000, 10000, 10000, 10000, 10000]

    tfrecord_path = r'D:\DP_CNN_Domca\tfrecordy'
    seed_value = 42

    for k in range(11):
        name = 'dataset_' + str(k + 1)
        tfrecord_name = 'DP_train_SS_' + str(k + 1) + '.tfrecords'
        tfrecord_name2 = 'DP_train_SW_' + str(k + 1) + '.tfrecords'
        tfrecord_path_name = 'tfrecord_path_' + str(k + 1)
        # cesta k tfrecordum
        exec('%s = [os.path.join(tfrecord_path, tfrecord_name), os.path.join(tfrecord_path, tfrecord_name2)]'
             % (tfrecord_path_name))

        # vytvorim dataset z TFRecord souboru
        exec('%s = tf.contrib.data.TFRecordDataset(%s)' % (name, tfrecord_path_name))
        exec('%s = %s.map(_parse_function)' % (name, name))  # zisk dat z tfrecordu
        exec('%s = %s.shuffle(buffer_size=buffer_size_list[k], seed=seed_value)' % (name, name))  # random zamichani dat
        exec('%s = %s.repeat()' % (name, name))  # opakovani datasetu do nekonecna (pro uceni)

        # pro skupinky obrazku z datasetu pro vstup do jednotlivych iteraci uceni
        batch_name = 'batch_dataset_' + str(k + 1)
        exec('%s = %s.batch(batch_size)' % (batch_name, name))

        # tvorba iteratoru pro prochazeni datasetu po skupinach obr
        iterator_name = 'iterator_' + str(k + 1)
        exec('%s = %s.make_initializable_iterator()' % (iterator_name, batch_name))

        # pro zisk dalsi skupiny obr z datasetu
        next_element_name = 'next_element_' + str(k + 1)
        exec('%s = %s.get_next()' % (next_element_name, iterator_name))

    size_image = 128     # velikost obrazku
    num_channel = 1      # pocet barevnych kanalu obrazku (sedotonovy = 1)
    size_image_vec = size_image * size_image * num_channel    # velikost vektoru z obrazku
    shape_image = (size_image, size_image)      # pro zmenu velikosti pouzivanych poli

    """Testovaci data"""

with tf.name_scope("Validacni_data"):
    # Tvorba datasetu, ktery nacte examples z TFRecord a vezme roi a znaceni pro CNN
    num_class = 11  # pocet trid pro klasifikaci
    batch_size = 4  # pocet obr z 1 tridy pro 1 beh trenovani site

    for k in range(11):
        name = 'dataset_t' + str(k + 1)
        tfrecord_name = 'DP_test_SS_' + str(k + 1) + '.tfrecords'
        tfrecord_name2 = 'DP_test_SW_' + str(k + 1) + '.tfrecords'
        tfrecord_path_name = 'tfrecord_path_' + str(k + 1)
        # cesta k tfrecordum
        exec('%s = [os.path.join(tfrecord_path, tfrecord_name), os.path.join(tfrecord_path, tfrecord_name2)]'
             % (tfrecord_path_name))

        # vytvorim dataset z TFRecord souboru
        exec('%s = tf.contrib.data.TFRecordDataset(%s)' % (name, tfrecord_path_name))
        exec('%s = %s.map(_parse_function)' % (name, name))  # zisk dat z tfrecordu
        exec('%s = %s.shuffle(buffer_size=200, seed=seed_value)' % (name, name))  # random zamichani dat
        exec('%s = %s.repeat()' % (name, name))  # povoleni opakovani datasetu

        # pro skupinky obrazku z datasetu pro vstup do jednotlivych iteraci uceni
        batch_name = 'batch_dataset_t' + str(k + 1)
        exec('%s = %s.batch(batch_size)' % (batch_name, name))

        # tvorba iteratoru pro prochazeni datasetu po skupinach obr
        iterator_name = 'iterator_t' + str(k + 1)
        exec('%s = %s.make_initializable_iterator()' % (iterator_name, batch_name))

        # pro zisk dalsi skupiny obr z datasetu
        next_element_name = 'next_element_t' + str(k + 1)
        exec('%s = %s.get_next()' % (next_element_name, iterator_name))


def new_convolutional_layer(input, num_input_channels, size_filter, num_filter, pooling=True, name="Conv_layer"):
    """Funkce pro tvorbu nove konvolucni vrstvy.

    Vstupy:
    input = predchozi vrstva (4D tensor - 1. cislo obr 2. y-osa 3. x-osa 4. kanaly obr / filtry vrstvy)
    num_input_channels = pocet kanalu predchozi vrstvy
    size_filter = velikost filtru v nove vrstve: (size x size)
    num_filter = pocet filtru ve vrstve
    pooling = True -> nastavi pouziti max-poolingu pres oblasti 2x2
    Vystupy:
    layer = konvolucni vrstva = 4D tensor (jako vstupni vrstva)
    w = vahy filtru vrstvy
    """
    with tf.name_scope(name):
        shape = [size_filter, size_filter, num_input_channels, num_filter]  # tvar filtru pro konvoluci=vah (format TF)
        # tvorba pocatecnich vah (hodnot filtru)
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="Vahy")
        # tvorba pocatecniho prahu, 1 pro kazdy filtr
        b = tf.Variable(tf.constant(0.05, shape=[num_filter]), name="Prah")

        # 2D konvoluce
        layer = tf.nn.conv2d(input=input,
                             filter=w,
                             strides=[1, 1, 1, 1],  # krok = 1 ve vsech dimenzich (cislo obr, x osa, y osa, input-chan.)
                             padding='SAME')     # vstupni obraz bude oramovan nulami, aby mel vystup stejnou velikost
        layer += b     # pridani prahu k vysledku konvoluce -> hodnota pridana ke kazdemu kanalu filtru

        # Pooling vrstva
        if pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],      # 2x2 max-pooling
                                   strides=[1, 2, 2, 1],    # posun o 2 pixely po x i y
                                   padding='SAME')

        # ReLU vrstva - vypocita max(x, 0) pro kazdy pixel x
        layer = tf.nn.relu(layer)   # relu(max_pool(x)) == max_pool(relu(x)) -> ReLU az po poolingu -> uspora operaci
        return layer, w   # Vystup = vrstva a vahy filtru (pro zobrazeni vah)


def flatten_layer(layer):
    """Funkce pro upravu vrstvy (tensoru) pro vstup do plne propojenou vrstvu (4D->2D tensor).

    Vstup:
    layer = vrstva
    Vystupy:
    layer_flat = upravena vrstva
    num_features = pocet priznaku
    """
    layer_shape = layer.get_shape()     # tvar (rozmery) vstupni vrstvy (4D tensor)
    num_features = layer_shape[1:4].num_elements()  # (pocet priznaku = vyska obr * sirka obr * pocet kanalu)
    layer_flat = tf.reshape(layer, [-1, num_features])     # zmena tvaru vrstvy -> [pocet obr, pocet priznaku]
    # pocet obr = -1 -> spocita se
    return layer_flat, num_features     # Vystup = upravena vrstva a pocet priznaku


def new_fully_layer(input,          # predchozi vrstva
                    num_inputs,     # pocet vstupnich neuronu z predchozi vrstvy
                    num_outputs,    # pocet vystupnich neuronu
                    relu=True,
                    name="Fully_layer"):    # pouzit ReLU vrstvu ?
    """Funkce pro tvorbu plne propojene vrstvy.

    Vstupy:
    input = predchozi vrstva = 2D tensor [pocet obr, pocet vstupu]
    num_inputs = pocet vstupnich neuronu
    num_outputs = pocet vystupnich neuronu
    relu = True -> provede se i ReLU vrstva
    Vystup:
    layer = plne propojena vrstva
    """
    with tf.name_scope(name):
        shape = [num_inputs, num_outputs]   # velikost novych vah
        # vahy - normalni rozlozeni se smer. odch. 0.05
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="Vahy")
        # nove prahy (vektor delky = pocet vystupu)
        b = tf.Variable(tf.constant(0.05, shape=[num_outputs]), name="Prah")
        layer = tf.matmul(input, w) + b  # vypocet vrstvy nasobenim matic vstupu a vah + pricteni hodnoty prahu

        if relu:  # ReLU vrstva
            layer = tf.nn.relu(layer)

        return layer    # Vystup = plne propojena vrstva


""" Placeholdery - pro vstupy"""
x = tf.placeholder(tf.float32, shape=[None, size_image_vec], name="Vstup")  # pro vstupni obrazky
x_image = tf.reshape(x, [-1, size_image, size_image, num_channel])   # zmane velikosti -> 4D tensor
y = tf.placeholder(tf.float32, shape=[None, num_class], name="Znaceni")  # pro znaceni obrazku
y_class = tf.argmax(y, axis=1)   # pro zisk cisel = znaceni

""" Tvorba architektury site """
# 1. konvolucni vrstva
layer_conv1, weights_conv1 = new_convolutional_layer(input=x_image,
                                                     num_input_channels=num_channel,
                                                     size_filter=size_filter1,
                                                     num_filter=num_filter1,
                                                     pooling=True,
                                                     name="Konvolucni_1")
# 2. konvolucni vrstva
layer_conv2, weights_conv2 = new_convolutional_layer(input=layer_conv1,
                                                     num_input_channels=num_filter1,
                                                     size_filter=size_filter2,
                                                     num_filter=num_filter2,
                                                     pooling=True,
                                                     name="Konvolucni_2")
# 3. konvolucni vrstva
layer_conv3, weights_conv3 = new_convolutional_layer(input=layer_conv2,
                                                     num_input_channels=num_filter2,
                                                     size_filter=size_filter3,
                                                     num_filter=num_filter3,
                                                     pooling=False,
                                                     name="Konvolucni_3")
# 4. konvolucni vrstva
layer_conv4, weights_conv4 = new_convolutional_layer(input=layer_conv3,
                                                     num_input_channels=num_filter3,
                                                     size_filter=size_filter4,
                                                     num_filter=num_filter4,
                                                     pooling=True,
                                                     name="Konvolucni_4")
# upravena vrstva do 1 rozmeru pro plne propojene vrstvy
layer_flat, num_features = flatten_layer(layer_conv4)
# 1. plne propojena vrstva
layer_fc1 = new_fully_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=size_fully1,
                            relu=True,
                            name="Plne_prop_1")
# 2. plne propojena vrstva
layer_fc2 = new_fully_layer(input=layer_fc1,
                            num_inputs=size_fully1,
                            num_outputs=num_class,
                            relu=False,
                            name="Plne_prop_2")

""" Predikce trid """
counter = 0  # pro zjisteni, jak dlouho uz se nezlepsil vysledek
max_test_accuracy = 0  # pro zjisteni lepsiho vysledku -> ukladani site
y_pred = tf.nn.softmax(layer_fc2)  # vysledna predikce zarazeni do trid
y_pred_class = tf.argmax(y_pred, axis=1)  # vysledna trida, jako maximum z predikce
y_class_value = tf.reduce_max(y_pred, axis=1)   # pro zisk hodnoty "psti" zarazeni

""" Optimalizace """
with tf.name_scope("Krizova_entropie"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y)
    cost = tf.reduce_mean(cross_entropy)
    tf.summary.scalar("Krizova_entropie", cost)

with tf.name_scope("Trenovani"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)   # cost funkce pro optimalizaci

with tf.name_scope("Presnost"):
    correct_prediction = tf.equal(y_pred_class, y_class)       # bool s true, kde se tridy rovnaji
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # presnost klasifikace site
    tf.summary.scalar("Presnost", accuracy)

summ = tf.summary.merge_all()  # pro tensorboard

""" TensorFlow Run Session"""
session = tf.Session()  # tvorba TensorFlow session
session.run(tf.global_variables_initializer())  # inicializace vsech promennych
session.run(iterator_1.initializer)
session.run(iterator_2.initializer)
session.run(iterator_3.initializer)
session.run(iterator_4.initializer)
session.run(iterator_5.initializer)
session.run(iterator_6.initializer)
session.run(iterator_7.initializer)
session.run(iterator_8.initializer)
session.run(iterator_9.initializer)
session.run(iterator_10.initializer)
session.run(iterator_11.initializer)

total_iterations = 0


def optimize(num_iterations):
    """Funkce pro provedeni optimalizace.

    Vstup: num_iterations = pocet iteraci optimalizace
    """
    global total_iterations  # pocet iteraci = glabalni promenna
    start_time = time.time()    # pocatecni cas pro vypis trvani optimalizace

    for i in range(total_iterations, total_iterations + num_iterations):
        x_batch_1, y_batch_1 = session.run(next_element_1)  # vezme dalsi skupinu obr pro trenovani
        x_batch_2, y_batch_2 = session.run(next_element_2)  # x_batch = skupina obrazku, y_batch = znaceni
        x_batch_3, y_batch_3 = session.run(next_element_3)
        x_batch_4, y_batch_4 = session.run(next_element_4)
        x_batch_5, y_batch_5 = session.run(next_element_5)
        x_batch_6, y_batch_6 = session.run(next_element_6)
        x_batch_7, y_batch_7 = session.run(next_element_7)
        x_batch_8, y_batch_8 = session.run(next_element_8)
        x_batch_9, y_batch_9 = session.run(next_element_9)
        x_batch_10, y_batch_10 = session.run(next_element_10)
        x_batch_11, y_batch_11 = session.run(next_element_11)
        x_batch = np.concatenate((x_batch_1, x_batch_2, x_batch_3, x_batch_4, x_batch_5, x_batch_6,
                                  x_batch_7, x_batch_8, x_batch_9, x_batch_10, x_batch_11), axis=0)
        y_batch = np.concatenate((y_batch_1, y_batch_2, y_batch_3, y_batch_4, y_batch_5, y_batch_6,
                                  y_batch_7, y_batch_8, y_batch_9, y_batch_10, y_batch_11), axis=0)

        feed_dict_train = {x: x_batch, y: y_batch}
        session.run(optimizer, feed_dict=feed_dict_train)   # spusti optimizer na skupine trenovacich dat
        if (i % 5 == 0):
            acc = session.run(accuracy, feed_dict=feed_dict_train)  # vypocet presnosti
            excel_poz1 = int(i / 5) + 4

        if (i % 100 == 0) or (i == total_iterations + num_iterations - 1):    # Zobraz presnost na skupine obr + iteraci
            msg = "Iterace: {0:>11}         Presnost trenovani: {1:>10.2%}"
            print(msg.format(i + 1, acc))
            global counter
            print_test_accuracy(iter=i, excel_poz=excel_poz1)

            if counter == 50:  # pokud nedoslo ke zlepseni po 5000 iteraci -> ukonci uceni
                total_iterations = i
                msg2 = "Maximalni presnost: {0:>10.2%}"
                print(msg2.format(max_test_accuracy))
                break

    total_iterations += num_iterations


writer = tf.summary.FileWriter("tensorboard\graf_CNN")
writer.add_graph(session.graph)


def print_test_accuracy(iter, excel_poz):
    """Funkce pro zjisteni presnosti CNN na validacnim datasetu.

    Vstupy:
    iter = iterace optimalizace
    excel_poz = pozice v excel souboru pro ulozeni hodnot
    """
    global num_class
    num_test = 2200  # Pocet obrazku testovaci sady
    class_pred = np.zeros(shape=num_test, dtype=np.int)  # pole pro predikovane tridy
    class_true = np.zeros(shape=num_test, dtype=np.int)  # pole pro oznacene tridy
    batch_test = 44

    # reinicilizace iteratoru (testovaci dataset prochazen vzdy od zacatku)
    session.run(iterator_t1.initializer)
    session.run(iterator_t2.initializer)
    session.run(iterator_t3.initializer)
    session.run(iterator_t4.initializer)
    session.run(iterator_t5.initializer)
    session.run(iterator_t6.initializer)
    session.run(iterator_t7.initializer)
    session.run(iterator_t8.initializer)
    session.run(iterator_t9.initializer)
    session.run(iterator_t10.initializer)
    session.run(iterator_t11.initializer)

    # Predikce trid po skupinach testovacich obrazku
    i = 0  # pocatecni index dalsi skupiny obr

    while i < num_test:
        j = min(i + batch_test, num_test)  # koncovy index dalsi skupiny obr

        x_test_1, y_test_1 = session.run(next_element_t1)  # vezme dalsi skupinu obr pro trenovani
        x_test_2, y_test_2 = session.run(next_element_t2)  # x_batch = skupina obrazku, y_batch = znaceni
        x_test_3, y_test_3 = session.run(next_element_t3)
        x_test_4, y_test_4 = session.run(next_element_t4)
        x_test_5, y_test_5 = session.run(next_element_t5)
        x_test_6, y_test_6 = session.run(next_element_t6)
        x_test_7, y_test_7 = session.run(next_element_t7)
        x_test_8, y_test_8 = session.run(next_element_t8)
        x_test_9, y_test_9 = session.run(next_element_t9)
        x_test_10, y_test_10 = session.run(next_element_t10)
        x_test_11, y_test_11 = session.run(next_element_t11)

        x_test = np.concatenate((x_test_1, x_test_2, x_test_3, x_test_4, x_test_5, x_test_6,
                                 x_test_7, x_test_8, x_test_9, x_test_10, x_test_11), axis=0)
        y_test = np.concatenate((y_test_1, y_test_2, y_test_3, y_test_4, y_test_5, y_test_6,
                                 y_test_7, y_test_8, y_test_9, y_test_10, y_test_11), axis=0)

        feed_dict_test = {x: x_test, y: y_test}    # da skupinu obrazku s jejich znacenim do dictionary
        class_pred[i:j] = session.run(y_pred_class, feed_dict=feed_dict_test)  # zisk predikovanych trid
        class_true[i:j] = session.run(y_class, feed_dict=feed_dict_test)  # oznacene tridy

        i = j

    correct = (class_true == class_pred)  # boolean pole - zda je obr spravne klasifikovan
    correct_sum = correct.sum()  # pocet spravne klasifikovanych obrazku
    test_accuracy = (correct_sum) / num_test  # vypocet presnosti klasifikace (pocet spravnych / celkovy pocet)

    excel_poz2 = int(iter / 5) + 4
    global max_test_accuracy
    global counter

    cm = count_statistics(class_true, class_pred, excel_poz)

    if (test_accuracy > max_test_accuracy) and (test_accuracy > 0.5):  # zlepsila se presnost na testovaci sade -> uloz
        max_test_accuracy = test_accuracy
        counter = 0

    elif (test_accuracy < max_test_accuracy):
        counter = counter + 1


def count_statistics(class_true, class_pred, excel_poz):
    """Funkce pro vypocet a zaznam statistickych hodnot pri uceni CNN.

    Vstupy:
    class_true = oznacene tridy
    class_pred = tridy predikovane CN
    excel_poz = pozice v excel souboru pro ulozeni hodnot
    """
    cm = confusion_matrix(y_true=class_true, y_pred=class_pred)  # matice chyb
    global num_class
    TP = np.zeros((num_class,), dtype=int)  # spravne pozitivni
    FN = np.zeros((num_class,), dtype=int)  # falesne negativni
    FP = np.zeros((num_class,), dtype=int)  # falesne pozitivni
    TN = cm[num_class - 1, num_class - 1]  # spravne negativni (pozadi)
    TN_2 = np.zeros((num_class,), dtype=int)  # spravne negativni

    for i in range(num_class):
        for j in range(num_class):
            if i == j:
                TP[i] = cm[i, j]
            else:
                FN[i] = FN[i] + cm[i, j]
                FP[i] = FP[i] + cm[j, i]

        TN_2[i] = sum(sum(cm)) - TP[i] - FN[i] - FP[i]

    TP_all = sum(TP[0:10])
    FN_all = sum(FN[0:10])
    FP_all = sum(FP[0:10])
    TN_all = np.mean(TN_2)

    sensitivity = TP_all / (TP_all + FN_all)  # sensitivita
    specificity = TN / (TN + FP_all)  # specificita
    specificity_2 = TN_all / (TN_all + FP_all)  # specificita (pro tridy)
    PPV = TP_all / (TP_all + FP_all)  # pozitivni prediktivni hodnota
    NPV = TN / (TN + FN_all)  # negativni prediktivni hodnota
    NPV_2 = TN_all / (TN_all + FN_all)  # negativni prediktivni hodnota (pro tridy)
    return cm


optimize(num_iterations=30000)  # spusteni optimalizace

r"""tensorboard --logdir=C:\Users\Domca\Desktop\DP_CNN\DP_kod\tensorboard\graf_CNN_22_SW,SS"""
