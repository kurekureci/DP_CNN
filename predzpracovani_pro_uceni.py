r"""Predzpracovani dat pro uceni CNN - Sliding Window (SW), Selective Search (SS), Intersection over Union (IoU).

-> ulozeni 11 souboru.tfrecord - kazdy pro 1 tridu protrenovani CNN
"""
import numpy as np
import cv2
import os
from lxml import etree
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util


def bb_IoU(bbA, bbB):  # vypocet Intersection over Union (IoU)
    """Funkce pro vypocet IoU mezi 2 bounding boxy.

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


def getKey(item):
    return item[4]


def create_example(bb_zn, image, image_name):
    """Funkce pro tvorbu formatu pro ulozeni oblasti do TFrecord formatu.

    Vstupy:
        bb_zn = souradnice bounding boxu a znaceni
        image = vstupni obrazek
        image_name = nazev obrazku
    """
    roi = np.array(image[bb_zn[1]:bb_zn[3], bb_zn[0]:bb_zn[2]])  # roi v sedotonovem vstupnim obrazku
    resized_roi = cv2.resize(roi, (128, 128))  # zmena velikost roi na velikost pro CNN
    label = int(bb_zn[4])
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
        'label': dataset_util.int64_feature(label),
        'image_name': dataset_util.bytes_feature(image_name_raw)}))

    return example


def left_or_right(bb_zn, image):
    """Funkce pro rozdeleni trid (plice, ledviny) na pravou a levou.

    Vstupy:
        bb_zn = souradnice bounding boxu a znaceni
        image = vstupni obrazek
    """
    bb_min = bb_zn[0]
    bb_max = bb_zn[2]
    image_middle = (image.shape[1]) / 2
    diff_1 = abs(bb_min - image_middle)
    diff_2 = abs(bb_max - image_middle)

    if diff_1 > diff_2:
        if bb_min < image_middle:
            position = 1  # na obrazku vlevo
        else:
            position = 2  # na obrazku vpravo
    else:
        if bb_max < image_middle:
            position = 1  # na obrazku vlevo
        else:
            position = 2  # na obrazku vpravo

    return position


if __name__ == '__main__':
    cv2.setUseOptimized(True)  # urychleni s pouzitim multithreads
    cv2.setNumThreads(4)
    num_roi = np.zeros((12,), dtype=np.int)  # pro vypsani, kolik je roi pro jednotlive tridy

    """Nacteni obrazku a oznacenych bb (parsovani xml)"""
    images_path = r'C:\Users\Domca\Desktop\DP_CNN\Data_DP\pat_08_0\output\DP_all_shuffle.txt'  # cesta k txt s nazvy obr
    data_dir = r'C:\Users\Domca\Desktop\DP_CNN\Data_DP\pat_08_0'  # cesta ke slozce s obr (uprava jasu pro SS)
    annotations_dir = r'C:\Users\Domca\Desktop\DP_CNN\Data_DP\pat_08_0\annotations_all'  # cesta k anotacim
    label_map_path = r'C:\Users\Domca\Desktop\DP_CNN\Data_DP\pat_08_0\dataDP_label_map.pbtxt'  # cesta k mape znaceni
    images_list = dataset_util.read_examples_list(images_path)
    label_map_dict = label_map_util.get_label_map_dict(label_map_path)
    image_save_dir = r'C:\Users\Domca\Desktop\DP_CNN\data'  # cesta k puvodnim obr

    method = "SS"  # SW pro sliding window, SS pro selective search

    # Otevreni souboru .tfrecord pro ukladani oblasti a znaceni
    tfrecord_path = r'C:\Users\Domca\Desktop\DP_CNN\Data_DP\pat_08_0\output'  # cesta k ulozenym tfrecord
    tfrecords_filename_1 = os.path.join(tfrecord_path, 'DP_train_SW_1.tfrecords')
    tfrecords_filename_2 = os.path.join(tfrecord_path, 'DP_train_SW_2.tfrecords')
    tfrecords_filename_3 = os.path.join(tfrecord_path, 'DP_train_SW_3.tfrecords')
    tfrecords_filename_4 = os.path.join(tfrecord_path, 'DP_train_SW_4.tfrecords')
    tfrecords_filename_5 = os.path.join(tfrecord_path, 'DP_train_SW_5.tfrecords')
    tfrecords_filename_6 = os.path.join(tfrecord_path, 'DP_train_SW_6.tfrecords')
    tfrecords_filename_7 = os.path.join(tfrecord_path, 'DP_train_SW_7.tfrecords')
    tfrecords_filename_8 = os.path.join(tfrecord_path, 'DP_train_SW_8.tfrecords')
    tfrecords_filename_9 = os.path.join(tfrecord_path, 'DP_train_SW_9.tfrecords')
    tfrecords_filename_10 = os.path.join(tfrecord_path, 'DP_train_SW_10.tfrecords')
    tfrecords_filename_11 = os.path.join(tfrecord_path, 'DP_test_SW_11.tfrecords')
    writer_1 = tf.python_io.TFRecordWriter(tfrecords_filename_1)
    writer_2 = tf.python_io.TFRecordWriter(tfrecords_filename_2)
    writer_3 = tf.python_io.TFRecordWriter(tfrecords_filename_3)
    writer_4 = tf.python_io.TFRecordWriter(tfrecords_filename_4)
    writer_5 = tf.python_io.TFRecordWriter(tfrecords_filename_5)
    writer_6 = tf.python_io.TFRecordWriter(tfrecords_filename_6)
    writer_7 = tf.python_io.TFRecordWriter(tfrecords_filename_7)
    writer_8 = tf.python_io.TFRecordWriter(tfrecords_filename_8)
    writer_9 = tf.python_io.TFRecordWriter(tfrecords_filename_9)
    writer_10 = tf.python_io.TFRecordWriter(tfrecords_filename_10)
    writer_11 = tf.python_io.TFRecordWriter(tfrecords_filename_11)

    for id, example in enumerate(images_list):  # cyklus pres obrazky s nazvy v train.txt
        path = os.path.join(annotations_dir, example + '.xml')  # cesta k xml se znacenim pro obrazek
        with tf.gfile.GFile(path, 'r') as fid:
            xml_str = fid.read()                # -> otevreni souboru (r=pro cteni)
        xml = etree.fromstring(xml_str)  # parsuje xml ze stringu, vrati root node
        # rekurzivni parsovani xml obsahu do dictionary v pythonu
        data = dataset_util.recursive_parse_xml_to_dict(xml)['annotation']
        img_name = data['filename']
        print(img_name)
        img_path = os.path.join(data_dir, img_name)
        img_input = cv2.imread(img_path)  # barevny obr pro selective search
        img_save_path = os.path.join(image_save_dir, img_name)
        img_save = cv2.imread(img_save_path, 0)  # sedotonovy obr pro ulozeni do tfrecord formatu
        h_img, w_img, cha_img = img_input.shape

        width = int(data['size']['width'])
        height = int(data['size']['height'])
        bbB_with_class = []
        classes_text = []

        if not data['object']:
            print("Nema znaceni objektu.")
            break

        for obj in data['object']:  # prochazi jednotliva znaceni bb v xml
            xmin = (int(obj['bndbox']['xmin']))
            ymin = (int(obj['bndbox']['ymin']))
            xmax = (int(obj['bndbox']['xmax']))
            ymax = (int(obj['bndbox']['ymax']))
            classes_text.append(obj['name'].encode('utf8'))
            classes = (label_map_dict[obj['name']])
            bbB_with_class.append([xmin, ymin, xmax, ymax, classes])  # prida bbB na konec listu

        if not bbB_with_class:  # pokud u obrazku nejsou pozadovana znaceni -> dalsi obrazek
            print("Neni pozadovane znaceni.")
            break

        if method == "SW":
            """Sliding window"""
            # 12 velikosti posuvneho okna
            width = [30, 40, 40, 60, 60, 80, 80, 100, 100, 150, 150, 200]
            height = [30, 40, 60, 60, 80, 80, 100, 100, 120, 150, 200, 250]
            windowSize = (width, height)  # velikost okna
            step = 5   # krok okna
            rects = []

            for i, it in enumerate(width):
                windowSize = (width[i], height[i])  # menim velikost okna

                for y in range(0, img_input.shape[0], step):
                    for x in range(0, img_input.shape[1], step):
                        window = img_input[y:y + windowSize[1], x:x + windowSize[0]]

                        if window.shape[0] != height[i] or window.shape[1] != width[i]:  # okno nema poz. vel. pryc
                            continue
                        rects.append([x, y, x + width[i], y + height[i]])  # ulozeni pozice okna

            candidates = rects

        elif method == "SS":
            """Selective search"""
            ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # vytvori SSS. objekt
            ss.setBaseImage(img_input)  # nastaveni vstup. obr pro segmentaci
            method2 = 'q'  # q = quality / f = fast - pro zmeno metody Selective Search

            if (method2 == 'f'):  # fast - rychla, ale s mene vystupy
                ss.switchToSelectiveSearchFast()
            elif (method2 == 'q'):  # quality - pomala, ale s vice vystupy
                ss.switchToSelectiveSearchQuality()

            rects = ss.process()  # spusti selective search segmentaci obr

            candidates = []  # list pro vybrane oblasti vetsi nez mezni hodnota

            for i, rect in enumerate(rects):  # vyber a ulozeni vybranych oblasti
                x, y, w, h = rect
                rect_size = w * h
                if rect_size > 100:  # vynechani, pokud je velikost regionu mensi nez 200
                    candidates.append([x, y, x + w, y + h])

        """Kopie vstupniho obrazku pro zobrazeni"""
        imOut1 = img_input.copy()
        imOut2 = img_input.copy()
        imOut3 = img_input.copy()
        imOut4 = img_input.copy()
        imOut5 = img_input.copy()
        imOut6 = img_input.copy()
        imOut7 = img_input.copy()
        imOut8 = img_input.copy()
        imOut9 = img_input.copy()
        imOut10 = img_input.copy()
        imOut11 = img_input.copy()

        rect2 = []  # list pro oblasti do zpracovani CNN (s vhodnou hodnotou iou)
        labels = []

        """Znaceni oblasti dle IoU"""
        for id_bbA, bbA in enumerate(candidates):  # prochazim navrzene oblasti z SS pro vypocet IoU
            if (len(bbB_with_class) > 1):   # pokud je v 1 obr vice oznacenych objektu -> seradit
                sorted_bbB = sorted(bbB_with_class, key=getKey)  # serazeni dle znaceni (cisla)

                for id_bbB, bbB_zn in enumerate(sorted_bbB):   # prochazim znacene bounding boxy
                    bbB = bbB_zn[0:4]  # jen bounding box bez znaceni
                    iou = bb_IoU(bbA, bbB)

                    if (id_bbA == len(candidates) - 1):  # posledni bbox
                        rect2.append(bbB)  # pro trenovani pridam nakonec i anotovany bb
                        labels.append(bbB_zn[4])
                        bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]
                        example = create_example(bb_zn, img_save, img_name)

                        if labels[-1] == 1:
                            writer_1.write(example.SerializeToString())
                            num_roi[0] = num_roi[0] + 1
                            cv2.rectangle(imOut1, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        elif labels[-1] == 2:
                            writer_2.write(example.SerializeToString())
                            cv2.rectangle(imOut2, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[1] = num_roi[1] + 1
                        elif labels[-1] == 3:
                            position = left_or_right(bb_zn, img_input)

                            if position == 1:  # na obrazku vlevo
                                writer_3.write(example.SerializeToString())
                                cv2.rectangle(imOut3, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                              (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                                num_roi[2] = num_roi[2] + 1
                            else:
                                labels[-1] == 9
                                bb_zn[4] = 9
                                example = create_example(bb_zn, img_save, img_name)
                                writer_9.write(example.SerializeToString())
                                cv2.rectangle(imOut9, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                              (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                                num_roi[8] = num_roi[8] + 1

                        elif labels[-1] == 4:
                            writer_4.write(example.SerializeToString())
                            cv2.rectangle(imOut4, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[3] = num_roi[3] + 1
                        elif labels[-1] == 5:
                            writer_5.write(example.SerializeToString())
                            cv2.rectangle(imOut5, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[4] = num_roi[4] + 1
                        elif labels[-1] == 6:
                            writer_6.write(example.SerializeToString())
                            cv2.rectangle(imOut6, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[5] = num_roi[5] + 1
                        elif labels[-1] == 7:
                            position = left_or_right(bb_zn, img_input)

                            if position == 1:  # na obrazku vlevo
                                writer_7.write(example.SerializeToString())
                                cv2.rectangle(imOut7, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                              (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                                num_roi[6] = num_roi[6] + 1
                            else:
                                labels[-1] = 10
                                bb_zn[4] = 10
                                example = create_example(bb_zn, img_save, img_name)
                                writer_10.write(example.SerializeToString())
                                cv2.rectangle(imOut10, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                              (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                                num_roi[9] = num_roi[9] + 1

                        elif labels[-1] == 8:
                            writer_8.write(example.SerializeToString())
                            cv2.rectangle(imOut8, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[7] = num_roi[7] + 1

                    if (iou > 0.65) or ((iou > 0.6) and ((bbA[2] - bbA[0]) < 50)):  # oznaceni objektu pri IoU > 65%
                        rect2.append(bbA)
                        labels.append(bbB_zn[4])
                        bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]
                        example = create_example(bb_zn, img_save, img_name)

                        if labels[-1] == 1:
                            writer_1.write(example.SerializeToString())
                            cv2.rectangle(imOut1, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[0] = num_roi[0] + 1
                        elif labels[-1] == 2:
                            writer_2.write(example.SerializeToString())
                            cv2.rectangle(imOut2, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[1] = num_roi[1] + 1
                        elif labels[-1] == 3:
                            position = left_or_right(bb_zn, img_input)

                            if position == 1:  # na obrazku vlevo
                                writer_3.write(example.SerializeToString())
                                cv2.rectangle(imOut3, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                              (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                                num_roi[2] = num_roi[2] + 1
                            else:
                                labels[-1] = 9
                                bb_zn[4] = 9
                                example = create_example(bb_zn, img_save, img_name)
                                writer_9.write(example.SerializeToString())
                                cv2.rectangle(imOut9, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                              (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                                num_roi[8] = num_roi[8] + 1
                        elif labels[-1] == 4:
                            writer_4.write(example.SerializeToString())
                            cv2.rectangle(imOut4, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[3] = num_roi[3] + 1
                        elif labels[-1] == 5:
                            writer_5.write(example.SerializeToString())
                            cv2.rectangle(imOut5, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[4] = num_roi[4] + 1
                        elif labels[-1] == 6:
                            writer_6.write(example.SerializeToString())
                            cv2.rectangle(imOut6, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[5] = num_roi[5] + 1
                        elif labels[-1] == 7:
                            position = left_or_right(bb_zn, img_input)

                            if position == 1:  # na obrazku vlevo
                                writer_7.write(example.SerializeToString())
                                cv2.rectangle(imOut7, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                              (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                                num_roi[6] = num_roi[6] + 1
                            else:
                                labels[-1] = 10
                                bb_zn[4] = 10
                                example = create_example(bb_zn, img_save, img_name)
                                writer_10.write(example.SerializeToString())
                                cv2.rectangle(imOut10, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                              (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                                num_roi[9] = num_roi[9] + 1
                        elif labels[-1] == 8:
                            writer_8.write(example.SerializeToString())
                            cv2.rectangle(imOut8, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[7] = num_roi[7] + 1

                    elif (iou < 0.3 and iou > 0):  # oznaceni pozadi pri IoU < 30 %
                        rect2.append(bbA)
                        labels.append(11)  # oznaceni pozadi
                        bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]

                        if id_bbA % 1000 == 0:
                            example = create_example(bb_zn, img_save, img_name)
                            writer_11.write(example.SerializeToString())
                            num_roi[11] = num_roi[11] + 1

                        num_roi[10] = num_roi[10] + 1

                    if (id_bbB == len(sorted_bbB) - 1):  # u posledniho znaceni nebylo IoU > 65% -> pozadi
                        rect2.append(bbA)
                        labels.append(11)  # trida 11 = pozadi
                        bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]

                        if id_bbA % 1000 == 0:
                            example = create_example(bb_zn, img_save, img_name)
                            writer_11.write(example.SerializeToString())
                            num_roi[11] = num_roi[11] + 1

                        num_roi[10] = num_roi[10] + 1

            else:
                bbB_zn = bbB_with_class[0]  # znaceni 1 objektu
                bbB = bbB_zn[0:4]  # jen bounding box bez znaceni
                iou = bb_IoU(bbA, bbB)

                if (id_bbA == len(candidates) - 1):  # pro posledni bbox
                    rect2.append(bbB)  # pro trenovani pridam i anotovany bb
                    labels.append(bbB_zn[4])
                    bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]
                    example = create_example(bb_zn, img_save, img_name)

                    if labels[-1] == 1:
                        writer_1.write(example.SerializeToString())
                        num_roi[0] = num_roi[0] + 1
                        cv2.rectangle(imOut1, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                    elif labels[-1] == 2:
                        writer_2.write(example.SerializeToString())
                        cv2.rectangle(imOut2, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        num_roi[1] = num_roi[1] + 1
                    elif labels[-1] == 3:
                        position = left_or_right(bb_zn, img_input)

                        if position == 1:  # na obrazku vlevo
                            writer_3.write(example.SerializeToString())
                            cv2.rectangle(imOut3, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[2] = num_roi[2] + 1
                        else:
                            labels[-1] == 9
                            bb_zn[4] = 9
                            example = create_example(bb_zn, img_save, img_name)
                            writer_9.write(example.SerializeToString())
                            cv2.rectangle(imOut9, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[8] = num_roi[8] + 1
                    elif labels[-1] == 4:
                        writer_4.write(example.SerializeToString())
                        cv2.rectangle(imOut4, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        num_roi[3] = num_roi[3] + 1
                    elif labels[-1] == 5:
                        writer_5.write(example.SerializeToString())
                        cv2.rectangle(imOut5, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        num_roi[4] = num_roi[4] + 1
                    elif labels[-1] == 6:
                        writer_6.write(example.SerializeToString())
                        cv2.rectangle(imOut6, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        num_roi[5] = num_roi[5] + 1
                    elif labels[-1] == 7:
                        position = left_or_right(bb_zn, img_input)

                        if position == 1:  # na obrazku vlevo
                            writer_7.write(example.SerializeToString())
                            cv2.rectangle(imOut7, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[6] = num_roi[6] + 1
                        else:
                            labels[-1] == 10
                            bb_zn[4] = 10
                            example = create_example(bb_zn, img_save, img_name)
                            writer_10.write(example.SerializeToString())
                            cv2.rectangle(imOut10, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                          (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                            num_roi[9] = num_roi[9] + 1
                    elif labels[-1] == 8:
                        writer_8.write(example.SerializeToString())
                        cv2.rectangle(imOut8, (bbB[0], bbB[1]), (bbB[2], bbB[3]),
                                      (255, 0, 0), 1, cv2.LINE_AA)  # modry obdelnik
                        num_roi[7] = num_roi[7] + 1

                if (iou > 0.65) or ((iou > 0.6) and ((bbA[2] - bbA[0]) < 50)):  # oznaceni objektu pri IoU > 65%
                    print("sem tady!")
                    rect2.append(bbA)
                    labels.append(bbB_zn[4])
                    bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]
                    example = create_example(bb_zn, img_save, img_name)

                    if labels[-1] == 1:
                        writer_1.write(example.SerializeToString())
                        cv2.rectangle(imOut1, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[0] = num_roi[0] + 1
                    elif labels[-1] == 2:
                        writer_2.write(example.SerializeToString())
                        cv2.rectangle(imOut2, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[1] = num_roi[1] + 1
                    elif labels[-1] == 3:
                        position = left_or_right(bb_zn, img_input)

                        if position == 1:  # na obrazku vlevo
                            writer_3.write(example.SerializeToString())
                            cv2.rectangle(imOut3, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[2] = num_roi[2] + 1
                        else:
                            labels[-1] == 9
                            bb_zn[4] = 9
                            example = create_example(bb_zn, img_save, img_name)
                            writer_9.write(example.SerializeToString())
                            cv2.rectangle(imOut9, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[8] = num_roi[8] + 1
                    elif labels[-1] == 4:
                        writer_4.write(example.SerializeToString())
                        cv2.rectangle(imOut4, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[3] = num_roi[3] + 1
                    elif labels[-1] == 5:
                        writer_5.write(example.SerializeToString())
                        cv2.rectangle(imOut5, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[4] = num_roi[4] + 1
                    elif labels[-1] == 6:
                        writer_6.write(example.SerializeToString())
                        cv2.rectangle(imOut6, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[5] = num_roi[5] + 1
                    elif labels[-1] == 7:
                        position = left_or_right(bb_zn, img_input)

                        if position == 1:  # na obrazku vlevo
                            writer_7.write(example.SerializeToString())
                            cv2.rectangle(imOut7, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[6] = num_roi[6] + 1
                        else:
                            labels[-1] == 10
                            bb_zn[4] = 10
                            example = create_example(bb_zn, img_save, img_name)
                            writer_10.write(example.SerializeToString())
                            cv2.rectangle(imOut10, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                          (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                            num_roi[9] = num_roi[9] + 1
                    elif labels[-1] == 8:
                        writer_8.write(example.SerializeToString())
                        cv2.rectangle(imOut8, (bbA[0], bbA[1]), (bbA[2], bbA[3]),
                                      (0, 0, 255), 1, cv2.LINE_AA)  # cerveny obdelnik
                        num_roi[7] = num_roi[7] + 1

                elif(iou < 0.3 and iou > 0):  # oznaceni pozadi pri IoU < 30 %
                    rect2.append(bbA)
                    labels.append(11)  # oznaceni pozadi
                    bb_zn = [rect2[-1][0], rect2[-1][1], rect2[-1][2], rect2[-1][3], labels[-1]]

                    if id_bbA % 1000 == 0:
                        example = create_example(bb_zn, img_save, img_name)
                        writer_11.write(example.SerializeToString())
                        num_roi[11] = num_roi[11] + 1

                    num_roi[10] = num_roi[10] + 1

    print("spine, heart, lung, head, liver, aorta, kidney, spleen, lung2, kidney2, background, saved background")
    print(num_roi)

    for id_roi, roi in enumerate(num_roi):
        if roi > 0 and id_roi < 11:
            a = id_roi + 1
            a = str(a)
            im_name = "imOut" + a
            imOut = eval(im_name)
            cv2.imshow("Output", imOut)  # Zobrazeni
            cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Output", imOut.shape[1], imOut.shape[0])
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    writer_1.close()
    writer_2.close()
    writer_3.close()
    writer_4.close()
    writer_5.close()
    writer_6.close()
    writer_7.close()
    writer_8.close()
    writer_9.close()
    writer_10.close()
    writer_11.close()
