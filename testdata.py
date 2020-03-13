import os
filename_input = "sense_train.txt"
with open(filename_input) as f:
    lines = f.readlines()
folders = []
for line in lines:
    line = line.rstrip()
    folders.append(line)


def make_dataset():
    img = []
    ground = []
    for line in folders:
        img_list = [os.path.splitext(f)[0] for f in os.listdir(line) if f.startswith('frame_')]
        line_train = [(os.path.join(line, img_name + '.png')) for img_name in img_list]
        line_ground = [(os.path.join(line[0:-6], 'mask', img_name + '.png')) for  img_name in img_list]
        img.append(img_list)
        ground.append(line_ground)
    return img, ground

t1,t2 = make_dataset()
print(t1)
print(t2)
