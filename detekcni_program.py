r"""Detekce anatomickych struktur v CT 2D rezech pomoci CNN.

1. detekce specifickych oblasti (Selective Search, Sliding Window)
2. klasifikace oblasti pomoci CNN
3. eliminace chybnych detekci

- spusteni v prikazove radce Anacondy: python detekcni_program.py --img_name=NAZEV_OBRAZKU.png
- NAZEV = nazev obrazku ve slozce se zdrojovym kodem
"""
import numpy as np
import cv2
import os
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

flags = tf.app.flags
flags.DEFINE_string('img_name', '', 'Nazev vstupniho snimku.')
FLAGS = flags.FLAGS


def getKey(item):
    return item[4]


def create_example(bb_zn, image, image_name):
    """Funkce pro tvorbu polozky pro ulozeni oblasti do TFrecord formatu.

    Vstupy:
        bb_zn = souradnice bounding boxu a znaceni
        image = vstupni obrazek
        image_name = nazev obrazku
    """
    roi = np.array(image[bb_zn[1]:bb_zn[3], bb_zn[0]:bb_zn[2]])  # oblast
    # zmena velikost roi na velikost pro CNN
    resized_roi = cv2.resize(roi, (128, 128))
    height = resized_roi.shape[0]
    width = resized_roi.shape[1]
    image = resized_roi.tostring()
    image_name_raw = image_name.encode('utf8')

    example = tf.train.Example(features=tf.train.Features(feature={
        'roi_poz1': dataset_util.int64_feature(bb_zn[0]),
        'roi_poz2': dataset_util.int64_feature(bb_zn[1]),
        'roi_poz3': dataset_util.int64_feature(bb_zn[2]),
        'roi_poz4': dataset_util.int64_feature(bb_zn[3]),
        'roi_height': dataset_util.int64_feature(height),
        'roi_width': dataset_util.int64_feature(width),
        'image': dataset_util.bytes_feature(image),
        'image_name': dataset_util.bytes_feature(image_name_raw)}))

    return example


save_name = 'oblasti'
method = "both"  # SW pro sliding window, SS pro selective search, both pro obe

cv2.setUseOptimized(True)  # urychleni s pouzitim multithreads
cv2.setNumThreads(4)

"""Nacteni obrazku"""
img_name = FLAGS.img_name
# cesta k mape znaceni
label_map_path = r'.\dataDP_label_map.pbtxt'
label_map_dict = label_map_util.get_label_map_dict(label_map_path)

# Otevreni souboru .tfrecord pro ukladani oblasti a znaceni
# cesta k ulozenym tfrecord
tfrecords_filename = save_name + '.tfrecords'
writer = tf.python_io.TFRecordWriter(tfrecords_filename)

# Nacteni snimku
img_path = os.path.join(".", img_name)
img_input = cv2.imread(img_path)
img_input2 = np.array(img_input)
minimum = np.min(img_input2)
maximum = np.max(img_input2)
img_input2 = (img_input2 - minimum) / (maximum - minimum)  # pro zobrazeni
copy1 = img_input2.copy()
img_save = cv2.imread(img_path, 0)  # pro ulozeni
h_img, w_img, cha_img = img_input.shape

if method == "SW" or method == "both":
    """Sliding window"""
    # 12 velikosti posuvneho okna
    width = [30, 40, 40, 60, 60, 80, 80, 100, 100, 150, 150, 200]
    height = [30, 40, 60, 60, 80, 80, 100, 100, 120, 150, 200, 250]
    windowSize = (width, height)  # velikost okna
    rects = []

    for i, it in enumerate(width):
        windowSize = (width[i], height[i])  # menim velikost okna

        if i == 0 or i == 1:
            step = 5  # u oken pro aortu krok posunu 5
        else:
            step = 10  # u ostatnich oken krok posunu 10

        for y in range(0, img_input.shape[0], step):
            for x in range(0, img_input.shape[1], step):
                window = img_input[y:y +
                                   windowSize[1], x:x + windowSize[0]]

                # pokud okno nema poz. vel. - pryc
                if window.shape[0] != height[i] or window.shape[1] != width[i]:
                    continue
                # Logika pro okna - dle velikosti prohledavaji jen cast prostoru
                if i == 0 or i == 1:  # okno pro aortu
                    if y < (img_input.shape[0] / 4) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore/dole
                        continue
                    # moc vlevo/vpravo
                    if x < (img_input.shape[1] / 4) or (x + windowSize[0]) > ((img_input.shape[1] / 4) * 3):
                        continue
                if i == 2:  # okno pro pater a ledviny
                    if y < (img_input.shape[0] / 2) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore/dole
                        continue
                    if x < (img_input.shape[1] / 10) or x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                if i == 3:  # okno pro pater, ledviny, kraje plic
                    if y < (img_input.shape[0] / 3) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore/dole
                        continue
                    if x < (img_input.shape[1] / 10) or x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                if i == 4:  # okno pro pater, ledviny a mini plice
                    if y < (img_input.shape[0] / 2) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore
                        continue
                    if x < (img_input.shape[1] / 10) or x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                if i == 5:  # okno pro male srdce, sirsi pater a ledviny
                    if y < (img_input.shape[0] / 4) or y > ((img_input.shape[0] / 5) * 4):  # moc nahore/dole
                        continue
                    if x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                if i == 6:  # okno pro pater,velke ledviny, slezinu
                    if y < (img_input.shape[0] / 3) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore/dole
                        continue
                if i == 7:  # okno pro sirokou pater, srdce, male plice
                    if y < (img_input.shape[0] / 5) or y > ((img_input.shape[0] / 4) * 3):  # moc nahore/dole
                        continue
                    if x < (img_input.shape[1] / 8) or x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                if i == 8:  # okno pro srdce, mensi plice, jatra, slezinu
                    # moc nahore/dole
                    if y < (img_input.shape[0] / 5) or (y + windowSize[1]) > ((img_input.shape[0] / 8) * 7):
                        continue
                if i == 9:  # okno pro srdce, jatra
                    # moc nahore/dole
                    if y < (img_input.shape[0] / 4) or (y + windowSize[1]) > ((img_input.shape[0] / 8) * 7):
                        continue
                    if (x + windowSize[0]) > ((img_input.shape[1] / 4) * 3):
                        continue
                if i == 10 or i == 11:  # okno pro hlavu, plice, jatra
                    if x < (img_input.shape[1] / 30) or x > ((img_input.shape[1] / 4) * 3):  # moc vlevo/vpravo
                        continue
                # ulozeni pozice okna
                rects.append([x, y, x + width[i], y + height[i]])

    candidates = rects

if method == "SS" or method == "both":
    """Selective search"""
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # vytvori SSS. objekt
    ss.setBaseImage(img_input)  # nastaveni vstup. obr pro segmentaci
    method2 = 'q'  # q = quality / f = fast - pro zmenu metody Selective Search

    if (method2 == 'f'):  # fast - rychla, ale s mene vystupy
        ss.switchToSelectiveSearchFast()
    elif (method2 == 'q'):  # quality - pomala, ale s vice vystupy
        ss.switchToSelectiveSearchQuality()

    rects = ss.process()  # výstup z selective search

    if method == "SS":  # vybran jen selective search
        candidates = []

    for i, rect in enumerate(rects):  # vyber oblasti s velikosti > 100
        x, y, w, h = rect
        rect_size = w * h
        if rect_size > 100:
            candidates.append([x, y, x + w, y + h])

"""Ulozeni navrhu oblasti do tfRecordu"""
for id_bb, bb in enumerate(candidates):
    example = create_example(bb, img_save, img_name)
    writer.write(example.SerializeToString())

writer.close()


"""Konvolucni neuronova sit"""

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


"""Vstupni data"""


def _parse_function_2(example_proto):  # pro zisk oblasti a jejich pozice
    features = {"image": tf.FixedLenFeature([], tf.string),
                "roi_poz1": tf.FixedLenFeature([], tf.int64),
                "roi_poz2": tf.FixedLenFeature([], tf.int64),
                "roi_poz3": tf.FixedLenFeature([], tf.int64),
                "roi_poz4": tf.FixedLenFeature([], tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    # prevod na float [0.0, 1.0]
    image = tf.cast(image, tf.float32) * (1. / 255)
    image.set_shape([128 * 128])
    roi_poz1 = tf.cast(parsed_features['roi_poz1'], tf.int64)
    roi_poz2 = tf.cast(parsed_features['roi_poz2'], tf.int64)
    roi_poz3 = tf.cast(parsed_features['roi_poz3'], tf.int64)
    roi_poz4 = tf.cast(parsed_features['roi_poz4'], tf.int64)
    roi_poz = [roi_poz1, roi_poz2, roi_poz3, roi_poz4]
    return image, roi_poz


with tf.name_scope("Vstupni_data"):
    # Tvorba datasetu, ktery nacte examples z TFRecord
    num_class = 11  # pocet trid pro klasifikaci
    batch_size = 20  # pocet obr pro 1 beh testovani site

    tfrecord_path_1 = tfrecords_filename
    dataset_t1 = tf.contrib.data.TFRecordDataset(
        tfrecord_path_1)  # vytvorim dataset z TFRecord souboru
    dataset_t1 = dataset_t1.map(_parse_function_2)
    dataset_t1 = dataset_t1.repeat()   # muze se projezdet dokola
    batch_dataset_t1 = dataset_t1.batch(batch_size)
    iterator_t1 = batch_dataset_t1.make_initializable_iterator()
    next_element_t1 = iterator_t1.get_next()

    size_image = 128     # velikost obrazku
    num_channel = 1      # pocet barevnych kanalu obrazku (sedotonovy = 1)
    # velikost vektoru z obrazku
    size_image_vec = size_image * size_image * num_channel
    # pro zmenu velikosti pouzivanych poli
    shape_image = (size_image, size_image)


def new_convolutional_layer(input, num_input_channels, size_filter, num_filter, pooling=True, name="Convolutional"):
    """Funkce pro tvorbu nove konvolucni vrstvy.

    Vstupy:
    input = vstupni vrstva (4D tensor - 1.cislo obr 2.y-osa 3.x-osa 4.kanaly obr/filtry vrstvy)
    num_input_channels = pocet kanalu vstupni vrstvy
    size_filter = velikost filtru -> (size x size)
    num_filter = pocet filtru
    pooling = True = pouziti max-poolingu pres oblasti 2x2
    Vystupy:
    layer = konvolucni vrstva (4D tensor)
    w = vahy filtru
    """
    with tf.name_scope(name):
        shape = [size_filter, size_filter, num_input_channels, num_filter]  # tvar filtru/vah (format TF)
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.05), name="W")  # pocatecni vah (hodnot filtru)
        b = tf.Variable(tf.constant(0.05, shape=[num_filter]), name="B")    # pocatecni prah (1 pro kazdy filtr)

        # 2D konvoluce
        layer = tf.nn.conv2d(input=input,
                             filter=w,
                             strides=[1, 1, 1, 1],  # kroky v dimenzich (cislo obr, x osa, y osa, vstupni kanaly)
                             padding='SAME')     # vstupni obraz oramovan nulami, aby mel vystup stejnou velikost
        layer += b     # pridani prahu k vysledku konvoluce

        # Pooling vrstva
        if pooling:
            layer = tf.nn.max_pool(value=layer,
                                   ksize=[1, 2, 2, 1],      # 2x2 max-pooling
                                   # posun o 2 pixely po x i y
                                   strides=[1, 2, 2, 1],
                                   padding='SAME')

        # ReLU vrstva (max(x, 0) pro kazdy pixel x)
        layer = tf.nn.relu(layer)  # relu(max_pool(x)) == max_pool(relu(x)) -> uspora operaci
        return layer, w


def flatten_layer(layer):
    """Funkce pro upravu vrstvy pro vstup do plne propojene vrstvy (4D->2D tensor).

    Vstup:
    layer = vstupni vrstva
    Vystupy:
    layer_flat = upravena vrstva
    num_features = pocet priznaku (neuronu pro 1 vstup)
    """
    layer_shape = layer.get_shape()     # rozmery vstupni vrstvy (4D tensor)
    num_features = layer_shape[1:4].num_elements()   # pocet priznaku
    layer_flat = tf.reshape(layer, [-1, num_features])  # zmena tvaru vrstvy (pocet vstupu = -1 -> spocita se)
    return layer_flat, num_features


def new_fully_layer(input, num_inputs, num_outputs, relu=True, name="Fully_connected"):
    """Funkce pro tvorbu plne propojene vrstvy.

    Vstupy:
    input = vstupni vrstva = 2D tensor [pocet obr, pocet vstupu]
    num_inputs = pocet vstupnich neuronu
    num_outputs = pocet vystupnich neuronu
    relu = True = i ReLU vrstva
    Vystup:
    layer = plne propojena vrstva
    """
    with tf.name_scope(name):
        shape = [num_inputs, num_outputs]   # velikost novych vah
        w = tf.Variable(tf.truncated_normal(shape, stddev=0.05))  # pocatecni vahy
        b = tf.Variable(tf.constant(0.05, shape=[num_outputs]))  # pocatecni prahy
        layer = tf.matmul(input, w) + b

        if relu:  # ReLU vrstva
            layer = tf.nn.relu(layer)

        return layer


""" Placeholdery - pro vstupy"""
x = tf.placeholder(tf.float32, shape=[None, size_image_vec], name="x")  # pro vstupni snimky
# zmane velikosti vstupu -> 4D tensor
x_image = tf.reshape(x, [-1, size_image, size_image, num_channel])
# pro znaceni snimku
y = tf.placeholder(tf.float32, shape=[None, num_class], name="labels")
y_class = tf.argmax(y, axis=1)   # pro zisk cisel = znaceni

""" Tvorba architektury site """
# 1. konvolucni vrstva
layer_conv1, weights_conv1 = new_convolutional_layer(input=x_image,
                                                     num_input_channels=num_channel,
                                                     size_filter=size_filter1,
                                                     num_filter=num_filter1,
                                                     pooling=True,
                                                     name="Convolutional_1")
# 2. konvolucni vrstva
layer_conv2, weights_conv2 = new_convolutional_layer(input=layer_conv1,
                                                     num_input_channels=num_filter1,
                                                     size_filter=size_filter2,
                                                     num_filter=num_filter2,
                                                     pooling=True,
                                                     name="Convolutional_2")
# 3. konvolucni vrstva
layer_conv3, weights_conv3 = new_convolutional_layer(input=layer_conv2,
                                                     num_input_channels=num_filter2,
                                                     size_filter=size_filter3,
                                                     num_filter=num_filter3,
                                                     pooling=False,
                                                     name="Convolutional_3")
# 4. konvolucni vrstva
layer_conv4, weights_conv4 = new_convolutional_layer(input=layer_conv3,
                                                     num_input_channels=num_filter3,
                                                     size_filter=size_filter4,
                                                     num_filter=num_filter4,
                                                     pooling=True,
                                                     name="Convolutional_4")
# upravena vrstva do 1 rozmeru pro plne propojene vrstvy
layer_flat, num_features = flatten_layer(layer_conv4)
# plne propojene vrstvy
layer_fc1 = new_fully_layer(input=layer_flat,
                            num_inputs=num_features,
                            num_outputs=size_fully1,
                            relu=True,
                            name="Fully_connect_1")
layer_fc2 = new_fully_layer(input=layer_fc1,
                            num_inputs=size_fully1,
                            num_outputs=num_class,
                            relu=False,
                            name="Fully_connect_2")

""" Predikce trid """
counter = 0  # pro zjisteni, jak dlouho uz se nezlepsil vysledek
max_test_accuracy = 0  # pro zjisteni lepsiho vysledku -> ukladani site
y_pred = tf.nn.softmax(layer_fc2)  # vysledna predikce zarazeni do trid
y_pred_class = tf.argmax(y_pred, axis=1)  # vysledna trida, jako maximum z predikce
y_class_value = tf.reduce_max(y_pred, axis=1)  # pro zisk hodnoty "psti" zarazeni

""" Optimalizace """
with tf.name_scope("Cross_entropy"):
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y)
    cost = tf.reduce_mean(cross_entropy)

with tf.name_scope("Train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)   # cost funkce pro optimalizaci

with tf.name_scope("Accuracy"):
    correct_prediction = tf.equal(y_pred_class, y_class)  # true, kde se tridy rovnaji
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # presnost klasifikace siti

""" TensorFlow Run Session"""
session = tf.Session()  # tvorba TensorFlow session
session.run(tf.global_variables_initializer())  # inicializace vsech promennych
session.run(iterator_t1.initializer)


def classify():
    """Funkce pro klasifikaci oblasti pomoci CNN.

    Vystup:
    class_pred = predikovane tridy
    class_value = vystupni hodnoty ze softmax funkce
    roi_position = pozice oblasti
    """
    global batch_size, candidates
    num_test = len(candidates)  # Pocet oblasti v testovaci sade
    num_test2 = (num_test - (num_test % batch_size)) + batch_size  # delka pro ukladani
    class_pred = np.zeros(shape=num_test2, dtype=np.int)  # pole pro predikovane tridy
    class_value = np.zeros(shape=num_test2, dtype=np.float32)  # pole pro hodnoty ze softmax fce
    roi_position = np.zeros((num_test2, 4), dtype=np.int)  # pole pro pozice bounding boxu

    # Predikce trid po skupinach oblasti testovacich obrazku
    i = 0  # pocatecni index dalsi skupiny obr

    while i < num_test:
        j = i + batch_size  # koncovy index dalsi skupiny obr
        x_test, roi_poz = session.run(next_element_t1)  # vezme dalsi skupinu obr pro trenovani
        y_test = np.zeros((1, 11), dtype=np.int)
        feed_dict_test = {x: x_test, y: y_test}
        class_pred[i:j] = session.run(y_pred_class, feed_dict=feed_dict_test)  # zisk predikovanych trid
        class_value[i:j] = session.run(y_class_value, feed_dict=feed_dict_test)  # zisk hodnoty predikce
        roi_position[i:j, :] = roi_poz
        i = j

    return class_pred[0:num_test], class_value[0:num_test], roi_position[0:num_test, :]


def optimistic_restore(session, save_file):
    """Pro obnoveni naucenych parametru site, ktere se vyskytuji i zde.

    Vstupy:
    session = nazev session
    save_file = cesta k ulozene CNN
    """
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    name2var = dict(zip(map(lambda x: x.name.split(
        ':')[0], tf.global_variables()), tf.global_variables()))
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = name2var[saved_var_name]
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


"""Nacteni ulozene CNN"""
model_path = r'.\Vysledna_CNN\saved_model'
optimistic_restore(session, model_path)

"""Klasifikace oblasti pomoci CNN"""
bboxy = np.array(candidates)
class_pred, class_value, roi_position = classify()  # klasifikace oblasti v CNN

"""Eliminace chybnych detekci"""


def bb_IoU(bbA, bbB):  # vypocet Intersection over Union (IoU)
    """Funkce pro vypocet prekryvu dle IoU mezi 2 bounding boxy.

    Vstupy: bbA, bbB = 2 bounding boxy
    """
    xmin = max(bbA[0], bbB[0])  # vypocet souradnic pro prusecik bb
    ymin = max(bbA[1], bbB[1])
    xmax = min(bbA[2], bbB[2])
    ymax = min(bbA[3], bbB[3])
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    inArea = w * h  # plocha pruseciku bb
    bbAArea = (bbA[2] - bbA[0] + 1) * (bbA[3] - bbA[1] + 1)  # plocha bb A
    bbBArea = (bbB[2] - bbB[0] + 1) * (bbB[3] - bbB[1] + 1)  # plocha bb B

    if (w <= 0 or h <= 0):  # pokud se bb A a B nekrizi
        iou = 0
    else:
        iou = inArea / float(bbAArea + bbBArea - inArea)  # podil plochy pruseciku a celkove plochy

    return iou


def non_max_overlap(boxes, threshold, class_value2, stop):
    """Funkce pro odstraneni prekryvajicich se bounding boxu (vysledku klasifikace oblasti).

        (s IoU > prah)
        Felzenszwalb et al.

    Vstupy: boxes = detekovane bounding boxy
            threshold = prah prekryvu oblasti
    """
    if len(boxes) == 0:  # if there are no boxes, return an empty list
        return []

    stop = min(stop, int(round(boxes.shape[0] / 2)))  # udava, kolik bude boxu na vystupu
    xmin = boxes[:, 0]  # coordinates of the bounding boxes
    ymin = boxes[:, 1]
    xmax = boxes[:, 2]
    ymax = boxes[:, 3]
    area = (xmax - xmin + 1) * (ymax - ymin + 1)  # area of BB

    twins = np.zeros((len(boxes),), dtype=np.int)
    pick = []

    for i1, i2 in enumerate(boxes):   # loop over all boxes
        for j1, j2 in enumerate(boxes):  # loop over boxes without the picked one (i)
            if j1 != i1:
                xx1 = max(xmin[i1], xmin[j1])
                yy1 = max(ymin[i1], ymin[j1])
                xx2 = min(xmax[i1], xmax[j1])
                yy2 = min(ymax[i1], ymax[j1])

                w = max(0, xx2 - xx1 + 1)  # width of interBB
                h = max(0, yy2 - yy1 + 1)  # height of interBB
                inArea = float(w * h)
                # overlap between the computed
                overlap = inArea / (area[j1] + area[i1] - inArea)

                if overlap > threshold:  # sufficient overlap -> suppress current BB
                    if class_value2[j1] < class_value2[i1]:
                        twins[i1] = twins[i1] + 1
                    else:
                        twins[j1] = twins[j1] + 1

    for i in range(0, stop):
        if twins.shape[0] > 0:
            poz = np.argmax(twins, axis=0)
            pick.append(boxes[poz, :])
            twins = np.delete(twins, poz, 0)  # delete indexes from index list

    pick = np.array(pick)
    return pick, twins  # vybran byl ten s nejvice prekryvy :D


def pick_the_best(boxes, class_value2, stop):
    """Funkce pro vybrani boxu s nej vystupni hodnotou ze softmax fce CNN.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
    """
    pick = []
    value = []
    stop = min(stop, int(round(boxes.shape[0] / 2)))  # udava, kolik bude boxu na vystupu

    for i in range(0, stop):
        if boxes.shape[0] > 0:
            poz = np.argmax(class_value2, axis=0)
            pick.append(boxes[poz, :])
            value.append(class_value2[poz])
            boxes = np.delete(boxes, poz, 0)  # delete indexes from index list
            class_value2 = np.delete(class_value2, poz, 0)

    pick = np.array(pick)

    return pick, value


def spine_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro pater.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if ymax < (pic_size[0] / 2):  # kdyz je ymax (radek) vys nez pulka vysky (vlevo)
            continue
        elif ymin < (pic_size[0] / 2):  # kdyz je ymin vys nez 1/2 vysky (vpravo)
            continue
        elif xmin < (pic_size[1] / 3):  # kdyz je xmin (sloupec) moc vlevo
            continue
        elif xmin > ((pic_size[1] / 3) * 2):  # kdyz je xmin  moc vpravo
            continue
        elif xmax < (pic_size[1] / 3):  # kdyz je xmax (sloupec) moc vlevo
            continue
        elif xmax > ((pic_size[1] / 3) * 2):  # kdyz je xmax (sloupec) moc vpravo
            continue
        elif (xmax - xmin) > (pic_size[1] / 4):  # kdyz je sirka > 1/4 obr
            continue
        elif (ymax - ymin) > (pic_size[0] / 4):  # kdyz je vyska > 1/4 obr
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def heart_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro srdce.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if ymin > (pic_size[0] / 2):  # kdyz je ymin > pulka (moc dole)
            continue
        elif ymax < (pic_size[0] / 4):  # kdyz je ymax < 1/4 (moc vysoko)
            continue
        elif ymax > ((pic_size[0] / 4) * 3):  # kdyz je ymax < 1/4 (moc dole)
            continue
        elif xmin < (pic_size[1] / 4):  # kdyz je xmin < 1/4 (moc vlevo)
            continue
        elif xmin > ((pic_size[1] / 4) * 3):  # kdyz je xmin > 3/4 (moc vpravo)
            continue
        elif xmax < (pic_size[1] / 4):  # kdyz je xmax < 1/4 (moc vlevo)
            continue
        elif xmax > ((pic_size[1] / 4) * 3):  # kdyz je xmax > 3/4 (moc vpravo)
            continue
        elif (xmax - xmin) > (pic_size[1] / 2.5):  # kdyz je sirka > 1/3 obr
            continue
        elif (ymax - ymin) > (pic_size[0] / 2.5):  # kdyz je vyska > 1/3 obr
            continue
        elif (xmax - xmin) < (pic_size[1] / 8):  # moc male
            continue
        elif (ymax - ymin) < (pic_size[0] / 8):  # moc male
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def left_lung_logic(boxes, class_value2, pic_size, img):
    """Funkce pro odstraneni spatnych boxu pro hlavu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
            img = detekovana oblast
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek
        roi = np.array(img[ymin:ymax, xmin:xmax])
        box_mean = np.mean(np.mean(roi))
        box_zeros = len(roi[np.where(roi == 0)])

        if xmax < (pic_size[1] / 3):  # kdyz je xmax moc vlevo
            continue
        elif xmax > ((pic_size[1] / 3) * 2):  # kdyz je xmax moc vpravo
            continue
        elif xmin > (pic_size[1] / 3):  # kdyz je xmin moc vpravo
            continue
        elif ymin > (pic_size[0] / 2):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif (xmax - xmin) < (pic_size[1] / 8):  # kdyz je sirka < 1/8 obr
            continue
        elif (ymax - ymin) < (pic_size[0] / 8):  # kdyz je vyska < 1/8 obr
            continue
        elif (xmax - xmin) > (pic_size[1] / 2):  # kdyz je sirka > 1/2 obr
            continue
        elif box_mean > 150:  # pokud je prumerny jas v obrazu > 50
            continue
        elif box_zeros < 20:  # pokud ma oblast plic malo nul
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def right_lung_logic(boxes, class_value2, pic_size, img):
    """Funkce pro odstraneni spatnych boxu pro hlavu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
            img = detekovana oblast
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek
        roi = np.array(img[ymin:ymax, xmin:xmax])
        box_mean = np.mean(np.mean(roi))
        box_zeros = len(roi[np.where(roi == 0)])

        if xmax < (pic_size[1] / 2):  # kdyz je xmax moc vlevo
            continue
        elif xmax > ((pic_size[1] / 8) * 7):  # kdyz je xmin moc vvlevo
            continue
        elif xmin < (pic_size[1] / 3):  # kdyz je xmin moc vvlevo
            continue
        elif xmin > ((pic_size[1] / 3) * 2):  # kdyz je xmin moc vvlevo
            continue
        elif ymin > (pic_size[0] / 2):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif (xmax - xmin) < (pic_size[1] / 8):  # kdyz je sirka < 1/8 obr
            continue
        elif (ymax - ymin) < (pic_size[0] / 8):  # kdyz je vyska < 1/8 obr
            continue
        elif (xmax - xmin) > (pic_size[1] / 2):  # kdyz je sirka > 1/2 obr
            continue
        elif box_mean > 150:  # pokud je prumerny jas v obrazu > 50
            continue
        elif box_zeros < 20:  # pokud ma oblast plic malo nul
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def head_logic(boxes, class_value2, pic_size, img):
    """Funkce pro odstraneni spatnych boxu pro hlavu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
            img = detekovana oblast
    """
    pick = []
    pick_70 = []
    value = []
    value_70 = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek
        roi = np.array(img[ymin:ymax, xmin:xmax])
        box_mean = np.mean(np.mean(roi))

        if xmax < (pic_size[1] / 2):  # kdyz je xmax < pulka (vlevo)
            continue
        elif xmin > (pic_size[1] / 2):  # kdyz je xmin > pulka (vpravo)
            continue
        elif (xmax - xmin) < (pic_size[1] / 4):  # kdyz je sirka < 1/4 obr
            continue
        elif (ymax - ymin) < (pic_size[0] / 2):  # kdyz je vyska < 1/2 obr
            continue
        elif (xmax - xmin) > (pic_size[1] / 2):  # kdyz je sirka > 1/2 obr
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

            if box_mean > 70:  # kdyz je prumerny jas > 70
                pick_70.append(boxes[i, :])
                value_70.append(class_value2[i])

    if len(pick_70) > 0:
        pick = np.array(pick_70)
        value = value_70
    else:
        pick = np.array(pick)

    return pick, value


def liver_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro jatra.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if xmax > ((pic_size[1] / 4) * 3):  # kdyz je xmax moc vpravo
            continue
        elif xmin > (pic_size[1] / 3):  # kdyz je xmin > pulka (vpravo)
            continue
        elif ymin > (pic_size[0] / 2):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif (xmax - xmin) > ((pic_size[1] / 5) * 4):  # kdyz je sirka < 4/5 obr
            continue
        elif (ymax - ymin) > ((pic_size[0] / 5) * 4):  # kdyz je vyska < 4/5 obr
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def aorta_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro aortu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if xmax > ((pic_size[1] / 3) * 2):  # kdyz je xmax moc vpravo
            continue
        elif xmax < (pic_size[1] / 3):  # kdyz je xmax moc vlevo
            continue
        elif xmin < (pic_size[1] / 3):  # kdyz je xmin moc vlevo
            continue
        elif xmin > ((pic_size[1] / 3) * 2):  # kdyz je xmin moc vpravo
            continue
        elif ymin < (pic_size[0] / 4):  # kdyz je ymin moc nahore
            continue
        elif ymin > ((pic_size[0] / 4) * 3):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 4):  # kdyz je ymax moc nahore
            continue
        elif ymax > ((pic_size[0] / 4) * 3):  # kdyz je ymax moc dole
            continue
        elif (xmax - xmin) > 40:  # kdyz je sirka > 40 px
            continue
        elif (ymax - ymin) > 40:  # kdyz je vyska > 40 px
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def left_kidney_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro levou ledvinu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if xmax > (pic_size[1] / 2):  # kdyz je xmax moc vpravo
            continue
        elif xmax < (pic_size[1] / 4):  # kdyz je xmax moc vlevo
            continue
        elif xmin < (pic_size[1] / 5):  # kdyz je xmin moc vlevo
            continue
        elif xmin > (pic_size[1] / 2):  # kdyz je xmin moc vpravo
            continue
        elif ymin < (pic_size[0] / 2):  # kdyz je ymin moc nahore
            continue
        elif ymin > ((pic_size[0] / 4) * 3):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif ymax > ((pic_size[0] / 8) * 7):  # kdyz je ymax moc dole
            continue
        elif (xmax - xmin) > (pic_size[1] / 4):  # kdyz je sirka > 1/4 obr
            continue
        elif (ymax - ymin) > (pic_size[0] / 4):  # kdyz je vyska > 1/4 obr
            continue
        elif (ymax - ymin) <= 30:  # kdyz je vyska < 30
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def right_kidney_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro levou ledvinu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if xmax < (pic_size[1] / 2):  # kdyz je xmax moc vlevo
            continue
        elif xmax > ((pic_size[1] / 4) * 3):  # kdyz je xmax moc vpravo
            continue
        elif xmin < (pic_size[1] / 2):  # kdyz je xmin moc vlevo
            continue
        elif xmin > ((pic_size[1] / 4) * 3):  # kdyz je xmin moc vpravo
            continue
        elif ymin < (pic_size[0] / 2):  # kdyz je ymin moc nahore
            continue
        elif ymin > ((pic_size[0] / 4) * 3):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif ymax > ((pic_size[0] / 8) * 7):  # kdyz je ymax moc dole
            continue
        elif (xmax - xmin) > (pic_size[1] / 4):  # kdyz je sirka > 1/4 obr
            continue
        elif (ymax - ymin) > (pic_size[0] / 4):  # kdyz je vyska > 1/4 obr
            continue
        elif (ymax - ymin) <= 30:  # kdyz je vyska < 30
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def spleen_logic(boxes, class_value2, pic_size):
    """Funkce pro odstraneni spatnych boxu pro slezinu.

    Vstupy: boxes = detekovane bounding boxy
            class_value2 = vystupni hodnoty ze softmax fce CNN pro boxy
            pic_size = velikost snimku
    """
    pick = []
    value = []

    for i, box in enumerate(boxes):
        xmin = box[0]  # min sloupec
        ymin = box[1]  # min radek
        xmax = box[2]  # max sloupec
        ymax = box[3]  # max radek

        if xmax < (pic_size[1] / 2):  # kdyz je xmax moc vlevo
            continue
        elif xmax > ((pic_size[1] / 8) * 7):  # kdyz je xmax moc vpravo
            continue
        elif xmin < (pic_size[1] / 2):  # kdyz je xmin moc vlevo
            continue
        elif xmin > ((pic_size[1] / 8) * 7):  # kdyz je xmin moc vpravo
            continue
        elif ymin < (pic_size[0] / 2):  # kdyz je ymin moc nahore
            continue
        elif ymin > ((pic_size[0] / 6) * 5):  # kdyz je ymin moc dole
            continue
        elif ymax < (pic_size[0] / 2):  # kdyz je ymax moc nahore
            continue
        elif ymax > ((pic_size[0] / 6) * 5):  # kdyz je ymax moc dole
            continue
        elif (xmax - xmin) > (pic_size[1] / 3):  # kdyz je sirka > 1/3 obr
            continue
        elif (ymax - ymin) > (pic_size[0] / 3):  # kdyz je vyska > 1/3 obr
            continue
        else:
            pick.append(boxes[i, :])
            value.append(class_value2[i])

    pick = np.array(pick)
    return pick, value


def make_average(boxes):
    """Funkce pro zprumerovani souradnic bounding boxu.

    Vstupy: boxes = detekovane bounding boxy 1 tridy
    """
    xmin = int(round(np.mean(boxes[:, 0])))
    ymin = int(round(np.mean(boxes[:, 1])))
    xmax = int(round(np.mean(boxes[:, 2])))
    ymax = int(round(np.mean(boxes[:, 3])))

    mean_box = [xmin, ymin, xmax, ymax]
    mean_box = np.array(mean_box)

    return mean_box


def make_maximum(boxes):
    """Funkce pro zprumerovani souradnic bounding boxu.

    Vstup: boxes = detekovane bounding boxy 1 tridy
    """
    xmin = int(round(min(boxes[:, 0])))
    ymin = int(round(min(boxes[:, 1])))
    xmax = int(round(max(boxes[:, 2])))
    ymax = int(round(max(boxes[:, 3])))

    mean_box = [xmin, ymin, xmax, ymax]
    mean_box = np.array(mean_box)

    return mean_box


image = img_input
height, width, _ = image.shape
font = cv2.FONT_HERSHEY_SIMPLEX

class_pred = np.array(class_pred)
class_value = np.array(class_value)

classes = ['Pater', 'Srdce', 'Leva plice', 'Hlava', 'Jatra', 'Aorta',
           'Leva ledvina', 'Slezina', 'Prava plice', 'Prava ledvina', 'Pozadi']

for i in range(1, 11):   # prochazeni jednotlivych trid
    bb_pick = (bboxy[np.where(class_pred == (i - 1)), :])   # vyber bboxu predikovanych pro tridu
    class_value_pick = (class_value[np.where(class_pred == (i - 1))])   # vyber hodnot softmax fce boxu

    if bb_pick.size:
        if bb_pick.shape[1] > 1:  # vice detekovanych bounding boxu pro 1 kategorii
            if i == 1:
                pick, value = spine_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 2:
                pick, value = heart_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 3:
                pick, value = left_lung_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]), image)
            elif i == 4:
                pick, value = head_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]), image)
            elif i == 5:
                pick, value = liver_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 6:
                pick, value = aorta_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 7:
                pick, value = left_kidney_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 8:
                pick, value = spleen_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))
            elif i == 9:
                pick, value = right_lung_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]), image)
            elif i == 10:
                pick, value = right_kidney_logic(bb_pick[0, :, :], class_value_pick, np.array([height, width]))

            if len(pick.shape) > 1:
                stop = 2
                pick1, value2 = pick_the_best(pick, value, stop)
                pick2, twins = non_max_overlap(pick, 0.5, value, stop)

                if len(pick1.shape) > 1:
                    pick3 = np.concatenate((pick1, pick2), axis=0)
                    pick3_iou_zeros = np.zeros((pick3.shape[0],), dtype=np.int)

                    for i1 in range(pick3.shape[0] - 1):

                        for i2 in range(i1 + 1, pick3.shape[0]):
                            pick3_iou = bb_IoU(pick3[i1, :], pick3[i2, :])

                            if pick3_iou == 0:
                                pick3_iou_zeros[i1] += 1
                                pick3_iou_zeros[i2] += 1

                    pick3 = (pick3[np.where(pick3_iou_zeros != 3), :])
                    pick3 = pick3[0, :, :]

                    if pick3.size:
                        if len(pick3_iou_zeros[np.where(pick3_iou_zeros == 2)]) == 4:
                            # pokud jsou 2 a 2 boxy spolu (a dohromady bez prekryvu) -> beru nej softmax vystupu (pick1)
                            if i == 2 or i == 6 or i == 7 or i == 10 or i == 3 or i == 9 or i == 5:
                                pick4 = make_average(pick1)
                            else:
                                pick4 = make_maximum(pick1)

                        else:
                            if i == 2 or i == 6 or i == 7 or i == 10 or i == 3 or i == 9 or i == 5:
                                pick4 = make_average(pick3)
                            else:
                                pick4 = make_maximum(pick3)

            else:
                pick3 = []
                pick3 = np.array(pick3)
                pick4 = []
                pick4 = np.array(pick4)

        else:
            pick4 = bb_pick[0, :, :]

        if pick4.size:
            cv2.rectangle(copy1, (pick4[0], pick4[1]), (pick4[2], pick4[3]),
                          (250, 250, 0), 1, cv2.LINE_AA)  # vysledky (modre)
            cv2.putText(copy1, classes[i - 1], (pick4[0], pick4[1] - 2),
                        font, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

        pick3 = []
        pick3 = np.array(pick3)
        pick4 = []
        pick4 = np.array(pick4)

"""Zobrazeni vysledku"""
name1 = img_name + "_vysledek_detekce"
cv2.imshow(name1, copy1)
cv2.waitKey(0)
