import argparse
import tensorflow as tf
import cv2 as cv
import os
import numpy as np


def getImageName(image):
    tmp = image.split('/')
    return str(tmp[tmp.__len__() - 1]).split('.')


def save_images(image, output, name, count=None):
    tmp = os.path.join(os.getcwd(), output, name[0])
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    if count != None:
        img_parth = os.path.join(os.getcwd(), '%s/%s_%i.%s' % (tmp, name[0], count, name[1]))
    else:
        img_parth = os.path.join(os.getcwd(), '%s/all_faces_%s.%s' % (tmp, name[0], name[1]))
    cv.imwrite(img_parth, image)


def load_frozen_graph(frozen_graph_filename):
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())


def object_detection(args):
    with tf.io.gfile.GFile(args.frozen, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        if args.tb:
            tf.import_graph_def(
                graph_def,
                input_map=None,
                return_elements=None,
                name="",
                op_dict=None,
                producer_op_list=None
            )
        else:
            tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = cv.imread(args.image)
        rows = img.shape[0]
        cols = img.shape[1]
        if args.tb:
            inp = cv.resize(img, (640, 480))
        else:
            inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB
        # Run the model
        if args.tb:
            print("TB\n")
            (np_pred_boxes, np_pred_confidences) = sess.run([sess.graph.get_tensor_by_name('add:0'),
                                                             sess.graph.get_tensor_by_name('Reshape_2:0')],
                                                            feed_dict={
                                                                'x_in:0': inp.reshape(inp.shape[0], inp.shape[1], 3)})

            boxes_r = np.reshape(np_pred_boxes, (-1,
                                                 15,
                                                 20,
                                                 1,
                                                 4))
            confidences_r = np.reshape(np_pred_confidences, (-1,
                                                             15,
                                                             20,
                                                             1,
                                                             2))
            j = 0
            cell_pix_size = 32
            x_scale = 480 / rows
            y_scale = 640 / cols
            inp = inp[:, :, [0, 1, 2]]
            i = 0
            inp = cv.cvtColor(inp, cv.COLOR_RGB2BGR)
            copy = inp.copy()
            for n in range(1):
                for y in range(15):
                    for x in range(20):
                        conf = np.max(confidences_r[0, y, x, n, 1:])
                        if (conf > 0.95):
                            i += 1
                            bbox = boxes_r[0, y, x, n, :]
                            abs_cx = (int(bbox[0]) + cell_pix_size + cell_pix_size * x)
                            abs_cy = (int(bbox[1]) + cell_pix_size + cell_pix_size * y)
                            w = bbox[2]
                            h = bbox[3]
                            tmp = copy[int(abs_cy - (h / 2)):int(abs_cy + (h / 2)),
                                  int(abs_cx - (w / 2)):int(abs_cx + (w / 2))]
                            cv.resize(tmp, (int(tmp.shape[0] * x_scale), int(tmp.shape[1] * y_scale)))
                            save_images(tmp, args.output, getImageName(args.image), i)
                            cv.rectangle(inp, (int((abs_cx - w / 2)), int((abs_cy - h / 2))),
                                         (int((abs_cx + w / 2)), int((abs_cy + h / 2))),
                                         (0, y, x), thickness=2)
            if args.overview:
                tmp = cv.resize(inp, (cols, rows))
                save_images(tmp, args.output, getImageName(args.image))
        else:
            print("API\n")
            out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                            sess.graph.get_tensor_by_name('detection_scores:0'),
                            sess.graph.get_tensor_by_name('detection_boxes:0'),
                            sess.graph.get_tensor_by_name('detection_classes:0')],
                           feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
            num_detections = int(out[0][0])
            for i in range(num_detections):
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[2][0][i]]
                if score > 0.1:
                    x = bbox[1] * cols
                    y = bbox[0] * rows
                    right = bbox[3] * cols
                    bottom = bbox[2] * rows
                    save_images(img[int(y):int(bottom), int(x):int(right)], args.output, getImageName(args.image), i)
                    cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
            if args.overview :
                save_images(img, args.output, getImageName(args.image))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='A tool to detect faces in images and cut out the detected faces. ')
    parser.add_argument("--image", required=True,  help='Set path to the image.')
    parser.add_argument("--frozen", required=True, help='Set path to the frozen graph.bp.')
    parser.add_argument("--output", default="output", help='Set path to the detected faces. By default the path is output.')
    parser.add_argument("--tb", type=bool, default=False,help='Must true if you use graphs from Tensorbox. By default it is false.')
    parser.add_argument("--overview",type=bool,default=True,help='False if you save only the faces. By default it is true. ')
    args = parser.parse_args()
    object_detection(args)
