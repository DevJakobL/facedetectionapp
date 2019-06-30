import argparse
import tensorflow as tf
import cv2 as cv
import os

def getImageName(image):
    tmp = image.split('/')
    return str(tmp[tmp.__len__() - 1]).split('.')


def save_images(image,output,name,count=None):
    tmp = os.path.join(os.getcwd(),output, name[0])
    if not os.path.exists(tmp):
        os.makedirs(tmp)
    if count != None:
        img_parth = os.path.join(os.getcwd(), '%s/%s_%i.%s' % (tmp, name[0], count, name[1]))
    else:
        img_parth = os.path.join(os.getcwd(), '%s/all_faces_%s.%s'%(tmp, name[0],name[1]))
    cv.imwrite(img_parth, image)


def object_detection(args):
    with tf.io.gfile.GFile(args.frozen, 'rb') as f:
        graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())

    with tf.compat.v1.Session() as sess:
        # Restore session
        sess.graph.as_default()
        tf.import_graph_def(graph_def, name='')

        # Read and preprocess an image.
        img = cv.imread(args.image)
        rows = img.shape[0]
        cols = img.shape[1]
        inp = cv.resize(img, (300, 300))
        inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

        # Run the model
        out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                        sess.graph.get_tensor_by_name('detection_scores:0'),
                        sess.graph.get_tensor_by_name('detection_boxes:0'),
                        sess.graph.get_tensor_by_name('detection_classes:0')],
                       feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

        # Visualize detected bounding boxes.
        num_detections = int(out[0][0])
        for i in range(num_detections):
            classId = int(out[3][0][i])
            score = float(out[1][0][i])
            bbox = [float(v) for v in out[2][0][i]]
            if score > 0.3:
                x = bbox[1] * cols
                y = bbox[0] * rows
                right = bbox[3] * cols
                bottom = bbox[2] * rows
                save_images(img[int(y):int(bottom), int(x):int(right)], args.output, getImageName(args.image),i)
                cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)
    save_images(img, args.output,getImageName(args.image))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument("--image",required=True)
    parser.add_argument("--frozen",required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    object_detection(args)
