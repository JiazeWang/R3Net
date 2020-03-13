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
        line_ground = [(os.path.join(line[0:-6], 'mask','mask_'+ img_name[6:] + '.png')) for  img_name in img_list]
        img = img + img_list
        ground = ground + line_ground
    return img, ground

t1,t2 = make_dataset()
print(t1)
print(t2)
print(len(t1), len(t2))
