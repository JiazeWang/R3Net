import numpy as np
import os

import torch
from PIL import Image
from torch.autograd import Variable
from torchvision import transforms
import datetime
from config import ecssd_path, hkuis_path, pascals_path, sod_path, dutomron_path
from misc import check_mkdir, crf_refine, AvgMeter, cal_precision_recall_mae, cal_fmeasure
from model0 import R3Net
import json
import os
import pdb
filename_input = "sense_test_new.txt"
with open(filename_input) as f:
    lines = f.readlines()
folders = []
for line in lines:
    line = line.rstrip()
    #items = line.split(' ')
    folders.append(line)
    #folders.append(items[0])


torch.manual_seed(2018)

# set which gpu to use
torch.cuda.set_device(1)

# the following two args specify the location of the file of trained model (pth extension)
# you should have the pth file in the folder './$ckpt_path$/$exp_name$'
ckpt_path = './ckptvgg'
exp_name = 'R3Net'

args = {
    'snapshot': '100000',  # your snapshot filename (exclude extension name)
    'crf_refine': True,  # whether to use crf to refine results
    'save_results': True  # whether to save the resulting masks
}

img_transform = transforms.Compose([
    transforms.Resize(300),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
img_convert =  transforms.Compose([
    transforms.Resize(300)
])
to_pil = transforms.ToPILImage()

to_test = {'ecssd': ecssd_path, 'hkuis': hkuis_path, 'pascal': pascals_path, 'sod': sod_path, 'dutomron': dutomron_path}


def main():
    net = R3Net().cuda()

    print 'load snapshot \'%s\' for testing' % args['snapshot']
    net.load_state_dict(torch.load(os.path.join(ckpt_path, exp_name, args['snapshot'] + '.pth'), map_location={'cuda:1':'cuda:6'}))
    net.eval()

    results = {}
    num = 0

    with torch.no_grad():

        for root in folders:
            newrootname = root.split('/')[-2]+'_'+ root.split('/')[-1]
            precision_record, recall_record, = [AvgMeter() for _ in range(256)], [AvgMeter() for _ in range(256)]
            mae_record = AvgMeter()

            if args['save_results']:
                #check_mkdir('%s_%s' % ("sensetime", "test1"))
                #check_mkdir(os.path.join(ckpt_path, exp_name, '(%s) %s_%s' % (exp_name, name, args['snapshot'])))
                check_mkdir(os.path.join(ckpt_path, exp_name, '%s_%s_%s' % (newrootname, exp_name, args['snapshot'])))
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.startswith('frame_')]
            num = num + 1
            print 'predicting for  %d' % (num)

            for idx, img_name in enumerate(img_list):
                s_time = datetime.datetime.now()
                #test_time = (new_time-start_time).seconds
                #print 'predicting for  %d / %d' % ( idx + 1, len(img_list))
                print 'predicting for %s: %d / %d' % (root[-5:], idx + 1, len(img_list))
                img = Image.open(os.path.join(root, img_name + '.png')).convert('RGB')
                img_var = Variable(img_transform(img).unsqueeze(0), volatile=True).cuda()
                imgnew = img_convert(img)
                #imgnew = img
                prediction = net(img_var)
                prediction = np.array(to_pil(prediction.data.squeeze(0).cpu()))
                m_time = datetime.datetime.now()
                #Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '%s_%s_%s' % (newrootname, exp_name, args['snapshot']), img_name+'_without_crf' + '.png'))
                if args['crf_refine']:
                    prediction = crf_refine(np.array(imgnew), prediction)
                e_time =datetime.datetime.now()
                #print 'infer:', (m_time-s_time).total_seconds(), 'total:', (e_time-s_time).total_seconds()

                gt = np.array(img_convert(Image.open(os.path.join(root[0:-6], 'mask', 'mask_'+img_name[6:] + '.png')).convert('L')))

                precision, recall, mae = cal_precision_recall_mae(prediction, gt)
                for pidx, pdata in enumerate(zip(precision, recall)):
                    p, r = pdata
                    precision_record[pidx].update(p)
                    recall_record[pidx].update(r)
                mae_record.update(mae)

                if args['save_results']:
                    Image.fromarray(prediction).save(os.path.join(ckpt_path, exp_name, '%s_%s_%s' % (newrootname, exp_name, args['snapshot']), img_name + '.png'))
            fmeasure = cal_fmeasure([precord.avg for precord in precision_record],
                                    [rrecord.avg for rrecord in recall_record])
            results[root[-10:-6]] = {'fmeasure': fmeasure, 'mae': mae_record.avg}

    print 'test results:'
    print results
    json.dump( results, open( "results_v1.json", 'w' ) )

if __name__ == '__main__':

    main()
