

# data_dir = r'C:\Users\dtrimina\Desktop\SegProject\database\irseg'
data_dir = '../database/irseg'

image_h = 480
image_w = 640

# Model cfg
n_classes = 9

model_name = "cccmodel"
pretrained = None

# Train cfg
imgs_per_gpu = 16
num_workers = 20

lr = 0.0005
weight_decay = 1e-5
epochs = 200

print_step = 100

