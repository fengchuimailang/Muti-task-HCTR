logdir = "./logdir"
# TODO root to be changed
root = "H:/柳博的空间/data/CASIA_mini/"
train_tfrecord = root + "train.tfrecords"
valid_tfrecord = root + "valid.tfrecords"
alphabet_path = root + "alphabet.txt"
train_dataset_path = root + "Train_Dgr/"
train_image_path = root + "train_img/"
valid_dataset_path = root + "test_Dgr/"
valid_image_path = root + "test_img/"

# TODO meaning?
PAD_ID = 0
GO_ID = 1
EOS_ID = 2

image_height = 128
image_max_width = 4000
label_max_len = 100

batch_size = 1
# TODO too big?
learning_rate = 0.001

total_steps = 99999999
show_step = 1
test_step = 500
simple_step = 1
