import tensorflow as tf
import numpy as np
import os
import re
import cv2
import struct
from PIL import Image


import config
from utils import read_alphabet

def resize_image(image, position):
    """
    将图片放缩到config.image_height的高度 （128）
    :param image: 图像
    :param position: 坐标list
    :return: 放缩后的图像，放缩后的坐标
    """

    def scale_position(positions, rate):
        for i in range(len(positions)):
            for j in range(len(position[i])):
                positions[i][j] = int(float(position[i][j]) / rate)
        return positions

    width, height = image.size
    rate_height = float(height) / float(config.image_height)
    position = scale_position(position, rate_height)
    new_width = int(float(width) / rate_height)
    new_height = config.image_height
    image = image.resize((new_width, new_height))
    return image, position


def listdir(root):
    """
    找到根目录下的所有文件
    :param root:
    :return:
    """
    filelist = []
    for dirpath, dirname, filename in os.walk(root):
        for filepath in filename:
            filelist.append(os.path.join(dirpath, filepath))
    return filelist


def load_data(root, save_img_path):
    """
    读取dgr格式文件,以生成器的形式返回一行文字的：图像, 坐标, 标签, 标签长
    :param root:
    :param save_img_path:
    :return:
    """
    file_num = 0
    filelist = listdir(root)
    for filepath in filelist:
        bin_data = open(filepath, "rb").read()  # dgr 二进制文件内容
        file_num += 1  # 下一文件
        filename = os.path.split(filepath)[1]  # 文件名
        print("第{0}个图片{1}正在转化".format(file_num, filename))
        offset = 0  # 偏移量
        fmt_header = "l8s"
        sizeofheader, format = struct.unpack_from(fmt_header, bin_data, offset)
        illu_len = sizeofheader - 36
        fmt_header = "=l8s" + str(illu_len) + "s20s2h3i"
        sizeofheader, format, illu, codetype, codelen, bits, img_h, img_w, line_num = struct.unpack_from(fmt_header,
                                                                                                         bin_data,
                                                                                                         offset)
        offset += struct.calcsize(fmt_header)
        error_flag = 0  # 若文本行存在错误label，则跳过这行
        i = 0  # 第i行
        while i < line_num:
            image = np.ones((img_h, img_w))
            image = image * 255
            line_word = ""
            position = np.zeros((config.label_max_len, 4), dtype=np.int32)

            fmt_line = "i"
            word_num, = struct.unpack_from(fmt_line, bin_data, offset)
            offset += struct.calcsize(fmt_line)

            line_left = 0
            line_right = 0
            line_top = 99999
            line_down = 0
            tmp_offset = offset
            error_flag = 0
            j = 0
            i += 1  # 下一行
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_y, top_left_x, H, W = struct.unpack_from(fmt_1, bin_data,
                                                                          offset)  # 每个字符标签、左上角顶点坐标、字符图像高、宽

                if j == 0:
                    line_left = top_left_x
                if j == word_num - 1:
                    line_right = top_left_x + W
                if top_left_y < line_top:
                    line_top = top_left_y
                if top_left_y + H > line_down:
                    line_down = top_left_y + H

                singal_word = str(label1.decode('gbk', 'ignore').strip(b'\x00'.decode()))
                line_word += singal_word  # 整行文字

                offset += struct.calcsize(fmt_1)

                image_size = H * W
                j += 1
                fmt_image = '=' + str(image_size) + 'B'
                images = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((H, W))
                try:
                    image[top_left_y:top_left_y + H, top_left_x:top_left_x + W] = images
                except:
                    print("文件名：{0}，第{1}行，第{2}个字,{3}".format(filename, i, j, singal_word))
                    print(top_left_y, top_left_x, H, W)
                    error_flag = 1
                # draw = ImageDraw.Draw(image)
                # draw.rectangle([(top_left_x1, top_left_y1),(top_left_x1+H, top_left_y1+W)], outline=(0, 255, 0, 255))
                # plt.imshow(images)
                # plt.show()
                offset += image_size

            if error_flag:  # 如果有错，跳过这一行
                continue
            '''保存position信息'''
            offset = tmp_offset
            j = 0
            position_num = 0
            while j < word_num:
                fmt_1 = '2s4h'
                label1, top_left_y, top_left_x, H, W = struct.unpack_from(fmt_1, bin_data,
                                                                          offset)  # 每个字符标签、左上角顶点坐标、字符图像高、宽

                singal_word = str(label1.decode('gbk', 'ignore').strip(b'\x00'.decode()))  # 解码单个字
                if not singal_word == "":
                    position[position_num][0] = top_left_y - line_top
                    position[position_num][1] = top_left_x - line_left
                    position[position_num][2] = H
                    position[position_num][3] = W
                    position_num += 1
                    # line_top:line_down + 1, line_left:line_right + 1
                image_size = H * W
                offset += struct.calcsize(fmt_1)
                j += 1
                offset += image_size
            if not len(line_word) == position_num:
                print(len(line_word), position_num)
            '''保存每行'''
            image_line = image[line_top:line_down + 1, line_left:line_right + 1]
            line_file = save_img_path + filename[:-4] + '-' + str(i) + '.jpg'
            # 中文路径不能用imwrite
            # cv2.imwrite(line_file, image_line)
            cv2.imencode('.jpg', image)[1].tofile(line_file)
            im = Image.open(line_file)
            yield im, position, line_word, len(line_word)





def create_tfrecord(train_save_path, dataset_path, save_img_path):
    print("Create tfrecord")
    idx2symbol, symbol2idx = read_alphabet(config.alphabet_path)
    print(symbol2idx)
    writer = tf.python_io.TFRecordWriter(train_save_path)
    out_of_label_max_length = 0
    for image, position, label, line_len in load_data(dataset_path, save_img_path):
        # 图像预处理，裁剪缩放
        image, position = resize_image(image, position)
        # label = re.sub('\|', ' ', label)
        # label = list(label.strip())
        label_list = list(label)
        # print('label', label)
        transed = False
        for i in range(len(label_list)):

            if label_list[i] not in idx2symbol:
                label_list[i] = '*'
                transed = True
        # 如果图像被转换，那就保存图片
        if transed:
            print("由于不在字母表被转换的图片label：{0},转化后的label:{1}".format(label, "".join(label_list)))

        label_list = [symbol2idx[s] for s in label_list]
        label_list.append(config.EOS_ID)

        label_list = np.array(label_list, np.int32)
        if label_list.shape[0] > config.label_max_len or label_list.shape[0] <= 0:
            out_of_label_max_length += 1
            continue
        image = np.array(image)
        # image norm
        image = 255 - image
        # print(image)
        # image = (image-np.min(image))/(np.max(image)-np.min(image))
        # print(image)
        position = np.array(position)
        # location (lfv,1)
        # position x,y,h,w
        lfv = int(config.image_max_width / 16)
        location = np.zeros((lfv), dtype=np.float32)
        classification = np.zeros((lfv), dtype=np.int32)
        detection = np.zeros((lfv, 4), dtype=np.float32)
        grid_left = -16
        grid_right = 0
        word_idx = 0
        # TODO to be promoted
        for j in range(lfv):
            grid_left += 16
            grid_right += 16
            center = position[word_idx][1] + position[word_idx][3] / 2  # 第word_idx个字的中心坐标
            if center >= grid_left and center < grid_right:
                idx = int(center / 16)
                location[idx] = 1
                classification[idx] = label_list[word_idx]
                # detection[0,1,2,3] 横坐标 纵坐标  水平长度比例，垂直高度比例
                detection[idx][0] = center
                detection[idx][1] = position[word_idx][0] + position[word_idx][2] / 2
                detection[idx][2] = position[word_idx][3] / config.image_height
                detection[idx][3] = position[word_idx][2] / config.image_height
                word_idx += 1
                if word_idx == label_list.shape[0]:
                    break

        _image_width = tf.train.Feature(int64_list=tf.train.Int64List(value=[image.shape[1]]))
        _image = tf.train.Feature(bytes_list=tf.train.BytesList(value=[image.tobytes()]))
        _label = [tf.train.Feature(int64_list=tf.train.Int64List(value=[tok])) for tok in label_list]
        # _label_length = tf.train.Feature(int64_list=tf.train.Int64List(value=[label_list.shape[0]]))
        _location = tf.train.Feature(bytes_list=tf.train.BytesList(value=[location.tobytes()]))
        _classification = tf.train.Feature(bytes_list=tf.train.BytesList(value=[classification.tobytes()]))
        _detection = tf.train.Feature(bytes_list=tf.train.BytesList(value=[detection.tobytes()]))
        # _position = tf.train.Feature(bytes_list=tf.train.BytesList(value=[position.tobytes()]))
        example = tf.train.SequenceExample(
            context=tf.train.Features(feature={
                'image_width': _image_width,
                'image': _image,
                # 'label_length': _label_length,
                'location': _location,
                'classification': _classification,
                'detection': _detection
            }),
            feature_lists=tf.train.FeatureLists(feature_list={
                'label': tf.train.FeatureList(feature=_label)
            })
        )
        writer.write(example.SerializeToString())
    writer.close()
    print("tfrecord file generated.")


if __name__ == '__main__':
    # save_alphabet(config.alphabet_path)
    create_tfrecord(config.train_tfrecord, config.train_dataset_path, config.train_image_path)
    create_tfrecord(config.valid_tfrecord, config.valid_dataset_path, config.valid_image_path)
