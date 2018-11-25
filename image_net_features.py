import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
import os
import argparse
import sys

def main(_):
  if not FLAGS.vid_dir:
  	tf.logging.error('Must set flag --vid_dir.')
  	return -1
  included_extensions = ['avi', 'mp4', 'mkv', 'mpeg']
  file_list = [fn for fn in os.listdir(FLAGS.vid_dir) if any(fn.endswith(ext) for ext in included_extensions)]
  module = hub.load_module_spec(FLAGS.tfhub_module)
  height,width = hub.get_expected_image_size(module)
  resized_input_tensor = tf.placeholder(tf.float32, [None, width, width, 3])
  m = hub.Module(module)
  feature_tensor = m(resized_input_tensor)
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init)
  for file_name in file_list:
    print('Processing file: '+file_name)
    vidcap = cv2.VideoCapture(FLAGS.vid_dir + '/' + file_name)
    success,image = vidcap.read()
    count = 0
    frames = []
    while success:
      frames.append(image)
      success,image = vidcap.read()
      count += 1
    print('Number of frames: ', count)
    img = [cv2.cvtColor(cv2.resize(frames[i], dsize = (height, width), interpolation = cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)/255 for i in range(len(frames))]
    feature_vector = sess.run(feature_tensor, feed_dict = {resized_input_tensor: img})
    txt_name = FLAGS.vid_dir + '/' + os.path.splitext(file_name)[0] + '.csv'
    np.savetxt(txt_name, feature_vector)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--vid_dir',
      type=str,
      default='./test',
      help='Path to folder of videos.'
  )
  parser.add_argument(
      '--tfhub_module',
      type=str,
      default=(
          'https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1'),
      help="""\
      Which TensorFlow Hub module to use.
      See https://github.com/tensorflow/hub/blob/r0.1/docs/modules/image.md
      for some publicly available ones.\
      """)
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)