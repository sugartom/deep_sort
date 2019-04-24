# vim: expandtab:ts=4:sw=4
import os
import errno
import argparse
import numpy as np
import cv2
import tensorflow as tf

# # Yitao-TLS-Begin
# import os
# import sys
# from tensorflow.python.saved_model import builder as saved_model_builder
# from tensorflow.python.saved_model import signature_constants
# from tensorflow.python.saved_model import signature_def_utils
# from tensorflow.python.saved_model import tag_constants
# from tensorflow.python.saved_model import utils
# from tensorflow.python.util import compat

# tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
# FLAGS = tf.app.flags.FLAGS

import grpc
from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.python.framework import tensor_util
# # Yitao-TLS-End


# def _run_in_batches(f, data_dict, out, batch_size):
#     data_len = len(out)
#     num_batches = int(data_len / batch_size)

#     s, e = 0, 0
#     for i in range(num_batches):
#         s, e = i * batch_size, (i + 1) * batch_size
#         batch_data_dict = {k: v[s:e] for k, v in data_dict.items()}
#         out[s:e] = f(batch_data_dict)
#     if e < len(out):
#         batch_data_dict = {k: v[e:] for k, v in data_dict.items()}
#         out[e:] = f(batch_data_dict)

# def _run_in_batches(sess, output_var, input_var, data_x, out, batch_size):
#     data_len = len(out)
#     num_batches = int(data_len / batch_size)

#     s, e = 0, 0
#     for i in range(num_batches):
#         s, e = i * batch_size, (i + 1) * batch_size
#         out[s:e] = sess.run(output_var, feed_dict = {input_var : data_x[s:e]})
#     if e < len(out):
#         out[e:] = sess.run(output_var, feed_dict = {input_var : data_x[e:]})

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, tuple(patch_shape[::-1]))
    return image


class ImageEncoder(object):

    def __init__(self, istub, input_name="images",
                 output_name="features"):
        # self.session = tf.Session()
        # with tf.gfile.GFile(checkpoint_filename, "rb") as file_handle:
        #     graph_def = tf.GraphDef()
        #     graph_def.ParseFromString(file_handle.read())
        # tf.import_graph_def(graph_def, name="net")
        # self.input_var = tf.get_default_graph().get_tensor_by_name(
        #     "net/%s:0" % input_name)
        # self.output_var = tf.get_default_graph().get_tensor_by_name(
        #     "net/%s:0" % output_name)

        # assert len(self.output_var.get_shape()) == 2
        # assert len(self.input_var.get_shape()) == 4
        # self.feature_dim = self.output_var.get_shape().as_list()[-1]
        # self.image_shape = self.input_var.get_shape().as_list()[1:]

        # print("self.feature_dim = %s" % str(self.feature_dim))
        # print("self.image_shape = %s" % str(self.image_shape))
        self.feature_dim = 128
        self.image_shape = [128, 64, 3]

        # ichannel = grpc.insecure_channel("localhost:8500")
        # self.istub = prediction_service_pb2_grpc.PredictionServiceStub(ichannel)
        self.istub = istub

    def tomHelper(self, myInput):
        self.internal_request = predict_pb2.PredictRequest()
        self.internal_request.model_spec.name = 'actdet_deepsort'
        self.internal_request.model_spec.signature_name = 'predict_images'
        self.internal_request.inputs['input'].CopyFrom(
            tf.contrib.util.make_tensor_proto(myInput, shape=myInput.shape))

        self.internal_result = self.istub.Predict(self.internal_request, 10.0)

        result_value = tensor_util.MakeNdarray(self.internal_result.outputs['output'])

        # print(result_value)

        return result_value

    def __call__(self, data_x, batch_size=32):
        # print("[Yitao] ImageEncoder()'s sess.run() is called...")
        out = np.zeros((len(data_x), self.feature_dim), np.float32)

        data_len = len(out)
        num_batches = int(data_len / batch_size)

        s, e = 0, 0
        for i in range(num_batches):
            s, e = i * batch_size, (i + 1) * batch_size
            # out[s:e] = self.session.run(self.output_var, feed_dict = {self.input_var : data_x[s:e]})
            out[s:e] = self.tomHelper(data_x[s:e])
        if e < len(out):
            # out[e:] = self.session.run(self.output_var, feed_dict = {self.input_var : data_x[e:]})
            out[e:] = self.tomHelper(data_x[e:])

        # if True:
        #     # Yitao-TLS-Begin
        #     init_op = tf.initialize_all_variables()
        #     self.session.run(init_op)

        #     export_path_base = "actdet_deepsort"
        #     export_path = os.path.join(
        #         compat.as_bytes(export_path_base),
        #         compat.as_bytes(str(FLAGS.model_version)))
        #     print('Exporting trained model to ', export_path)
        #     builder = saved_model_builder.SavedModelBuilder(export_path)

        #     tensor_info_x = tf.saved_model.utils.build_tensor_info(self.input_var)
        #     tensor_info_y = tf.saved_model.utils.build_tensor_info(self.output_var)

        #     prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
        #         inputs={'input': tensor_info_x},
        #         outputs={'output': tensor_info_y},
        #         method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

        #     legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
        #     builder.add_meta_graph_and_variables(
        #         self.session, [tf.saved_model.tag_constants.SERVING],
        #         signature_def_map={
        #             'predict_images':
        #                 prediction_signature,
        #         },
        #         legacy_init_op=legacy_init_op)

        #     builder.save()

        #     print('Done exporting!')
        #     # Yitao-TLS-End


        return out


def create_box_encoder(istub, input_name="images",
                       output_name="features", batch_size=32):
    image_encoder = ImageEncoder(istub, input_name, output_name)
    image_shape = image_encoder.image_shape

    def encoder(image, boxes):
        image_patches = []
        for box in boxes:
            patch = extract_image_patch(image, box, image_shape[:2])
            if patch is None:
                # print("WARNING: Failed to extract image patch: %s." % str(box))
                patch = np.random.uniform(
                    0., 255., image_shape).astype(np.uint8)
            image_patches.append(patch)
        image_patches = np.asarray(image_patches)
        return image_encoder(image_patches, batch_size)

    return encoder


def generate_detections(encoder, mot_dir, output_dir, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)

    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        detection_file = os.path.join(
            detection_dir, sequence, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)] for row, feature
                               in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument(
        "--model",
        default="resources/networks/mars-small128.pb",
        help="Path to freezed inference graph protobuf.")
    parser.add_argument(
        "--mot_dir", help="Path to MOTChallenge directory (train or test)",
        required=True)
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to "
        "standard MOT detections Directory structure should be the default "
        "MOTChallenge structure: [sequence]/det/det.txt", default=None)
    parser.add_argument(
        "--output_dir", help="Output directory. Will be created if it does not"
        " exist.", default="detections")
    return parser.parse_args()


def main():
    args = parse_args()
    encoder = create_box_encoder(args.model, batch_size=32)
    generate_detections(encoder, args.mot_dir, args.output_dir,
                        args.detection_dir)


if __name__ == "__main__":
    main()
