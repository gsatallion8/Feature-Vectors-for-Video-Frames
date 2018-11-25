# Feature-Vectors-for-Video-Frames
Dealing with videos for classification can be difficult as there's both features across space as well as across time. Using a CNN based feature extractor would extract the spacial features and let an RNN/ LSTM take care of the temporal features. This repo contains code to extract the spatial features using pretrained networks.

image_net_features.py computes the features in each frame of a video and output it all to a single csv file per video.

image_net_opticla_flow_features computes both the image features as well as features in the optical flow i.e., relative motion in the image.
