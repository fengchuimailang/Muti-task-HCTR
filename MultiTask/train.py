import tensorflow as tf
import os
import time
from PIL import Image
import numpy as np
import cv2

from MultiTask.net import Net
from MultiTask.utils import sigmoid
import MultiTask.config as config


def box2original(pre, i, lfv, H, W):
    ori = [0] * 4
    x = pre[0]
    y = pre[1]
    w = pre[2]
    h = pre[3]
    xb = (i + sigmoid(x)) * W / lfv
    yb = sigmoid(y) * H
    wb = w * H
    hb = sigmoid(h) * H
    ori[0] = yb
    ori[1] = xb
    ori[2] = yb + wb
    ori[3] = xb + hb
    return ori


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def train():
    g = Net(config)
    g.build_net()
    idx2symbol, symbol2idx = g.idx2symbol, g.symbol2idx
    sv = tf.train.Supervisor(graph=g.graph, logdir=g.config.logdir)
    cfg = tf.ConfigProto()
    cfg.gpu_options.allow_growth = True
    with sv.managed_session(config=cfg) as sess:
        Locloss = 0
        Claloss = 0
        Detloss = 0
        time_start = time.time()
        for step in range(1, config.total_steps):
            if step == 1 or step == 299999 or step == 399999:
                img, label, loc, cla, label_len = sess.run([g.x, g.label, g.location, g.classification],
                                                           {g.train_stage: True})
                print("label:", label[0])
                label_t = [idx2symbol[s] for s in label[0]]
                print("label:", label_t)
                print("loc:", loc[0])
                print("cla:", cla[0])
                print("img shape", img.shape)
            loss, loc_loss, cla_loss, det_loss, _ = sess.run([g.loss, g.loc_loss, g.cla_loss, g.det_loss, g.train_op],
                                                             {g.train_stage: True})
            Locloss += loc_loss
            Claloss += cla_loss
            Detloss += det_loss
            if step % config.show_step == 0:
                print("step=%d,loc loss=%f,cla loss=%f,dec loss=%f,最近config.show_step用时=%f s" % (
                    step, Locloss / config.show_step, Claloss / config.show_step, Detloss / config.show_step,
                    time.time() - time_start))
                Locloss = 0
                Claloss = 0
                Detloss = 0
                time_start = time.time()
            if step % config.simple_step == 0:
                label, loc, loc_p, loc_pre_t = sess.run([g.label, g.location, g.loc_p, g.loc_pre_t],
                                                        {g.train_stage: True})
                label_t = [idx2symbol[s] for s in label[0]]
                print("label:", label_t)
                print("loc res sigmoid", loc_pre_t[0])
                print("loc pre:", loc_p[0])

            if step % config.test_step == 9999999:
                x, label, loc, cla, loc_pre, cla_pre, det_pre = sess.run(
                    [g.x, g.label, g.location, g.classification, g.loc_p, g.cla_p, g.det_p], {g.train_stage: True})
                lfv = int(config.image_max_width / 16)
                print("loc_p.shape", loc_pre.shape)
                print("det_p.shape", det_pre.shape)
                print("cla_p.shape", cla_pre.shape)
                for i in range(config.batch_size):
                    label_t = [idx2symbol for s in label[i]]
                    print()
                    print("Example %d:" % (i))
                    print("label:", label_t)
                    loc_p = loc_pre[i]  # lfv
                    cla_p = cla_pre[i]  # (lfv,7356)
                    det_p = det_pre[i]  # lfv,4
                    print("loc_p.shape", loc_p.shape)
                    print("det_p.shape", det_p.shape)
                    print("cla_p.shape", cla_p.shape)
                    cla_p_idx = np.argmax(cla_p, axis=-1)
                    print("cla_p_idx", cla_p_idx)
                    cla_p_t = []
                    t = 0
                    for j in cla_p_idx:
                        cla_p_t.append(cla_p[t][j])
                        t += 1
                    cla_p = np.array(cla_p_t)
                    conf_bbox = 0.8 * loc_p + 0.2 * cla_p  # [lfv]
                    dets = np.ones((lfv, 5))
                    img_h, img_q = config.image_height, config.image_max_width
                    for j in range(lfv):
                        dets[j][4] = conf_bbox[j]
                        tmp_det = box2original(det_p[j], j, lfv, img_h, img_w)
                        dets[j][0] = tmp_det[0]
                        dets[j][1] = tmp_det[1]
                        dets[j][2] = tmp_det[2]
                        dets[j][3] = tmp_det[3]
                    keep = nms(dets, 0.5)  # lfv中保留的bbox
                    order_x = det_p[:1].argsort[::1]  # 按照x顺序排序，由小到大
                    label_pre = []
                    for k in order_x:
                        if k in keep:
                            label_pre.append(cla_p_idx[k])
                    print("label_pre", label_pre)
                    label_pre = [idx2symbol[s] for s in label_pre]
                    print("label pre:", label_pre)
                    print()


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
