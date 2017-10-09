train_init_file = 'data/train_init.txt'
val_file = 'data/test.txt'
learning_rate = 0.0001
num_epochs = 300
train_batch_size = 256
display_step = 100

filewriter_path = "tensorboard"
checkpoint_path = "checkpoints"

dropout_rate = 0.5
num_classes = 500
skip_layers = ['fc8']
train_layers = ['fc6', 'fc7', 'fc8']

rest_file = 'data/train_rest.txt'
pred_file = 'data/train_pred.txt'
test_batch_size = 337
dropout_rate = 0.5
checkpoint_path = "checkpoints"

selectALNUM = 5000
