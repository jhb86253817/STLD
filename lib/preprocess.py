import os, cv2
import sys
import hdf5storage
import numpy as np
import random
import json
random.seed(1234)
np.random.seed(1234)

def process_300w(root_folder, folder_name, image_name, label_name, target_size, scale):
    image_path = os.path.join(root_folder, folder_name, image_name)
    label_path = os.path.join(root_folder, folder_name, label_name)

    with open(label_path, 'r') as ff:
        anno = ff.readlines()[3:-1]
        anno = [x.strip().split() for x in anno]
        anno = [[int(float(x[0])), int(float(x[1]))] for x in anno]
        image = cv2.imread(image_path)
        image_height, image_width, _ = image.shape
        anno_x = [x[0] for x in anno]
        anno_y = [x[1] for x in anno]
        bbox_xmin = min(anno_x)
        bbox_ymin = min(anno_y)
        bbox_xmax = max(anno_x)
        bbox_ymax = max(anno_y)

        bbox_width = bbox_xmax - bbox_xmin + 1
        bbox_height = bbox_ymax - bbox_ymin + 1

        bbox_xmin -= (scale-1)/2*bbox_width
        bbox_ymin -= (scale-1)/2*bbox_height
        bbox_xmax += (scale-1)/2*bbox_width
        bbox_ymax += (scale-1)/2*bbox_height

        bbox_xmin = max(bbox_xmin, 0)
        bbox_ymin = max(bbox_ymin, 0)
        bbox_xmax = min(bbox_xmax, image_width-1)
        bbox_ymax = min(bbox_ymax, image_height-1)

        bbox_width = bbox_xmax - bbox_xmin + 1
        bbox_height = bbox_ymax - bbox_ymin + 1
        anno = [[(x-bbox_xmin)/bbox_width, (y-bbox_ymin)/bbox_height] for x,y in anno]

        image_crop = image[int(bbox_ymin):int(bbox_ymax)+1, int(bbox_xmin):int(bbox_xmax)+1, :]
        image_crop = cv2.resize(image_crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
        return image_crop, anno

def process_wflw(anno, target_size):
    image_name = anno[-1]
    image_path = os.path.join('..', 'data', 'WFLW', 'WFLW_images', image_name)
    image = cv2.imread(image_path)
    image_height, image_width, _ = image.shape
    lms = anno[:196]
    lms = [float(x) for x in lms]
    lms_x = lms[0::2]
    lms_y = lms[1::2]

    bbox = anno[196:200]
    bbox = [float(x) for x in bbox]
    attrs = anno[200:206]
    attrs = np.array([int(x) for x in attrs])
    bbox_xmin, bbox_ymin, bbox_xmax, bbox_ymax = bbox

    width = bbox_xmax - bbox_xmin
    height = bbox_ymax - bbox_ymin
    scale = 1.0 
    bbox_xmin -= width * (scale-1)/2
    bbox_ymin -= height * (scale-1)/2
    bbox_xmax += width * (scale-1)/2
    bbox_ymax += height * (scale-1)/2

    bbox_xmin = max(bbox_xmin, 0)
    bbox_ymin = max(bbox_ymin, 0)
    bbox_xmax = min(bbox_xmax, image_width-1)
    bbox_ymax = min(bbox_ymax, image_height-1)
    width = bbox_xmax - bbox_xmin + 1
    height = bbox_ymax - bbox_ymin + 1
    image_crop = image[int(bbox_ymin):int(bbox_ymax)+1, int(bbox_xmin):int(bbox_xmax)+1, :]
    image_crop = cv2.resize(image_crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    tmp1 = [bbox_xmin, bbox_ymin]*98
    tmp1 = np.array(tmp1)
    tmp2 = [width, height]*98
    tmp2 = np.array(tmp2)
    lms = np.array(lms) - tmp1
    lms = lms / tmp2
    lms = lms.tolist()
    lms = zip(lms[0::2], lms[1::2])
    return image_crop, list(lms)

def process_aflw(root_folder, image_name, bbox, anno, target_size):
    image = cv2.imread(os.path.join(root_folder, 'AFLW', 'flickr', image_name))
    image_height, image_width, _ = image.shape
    anno_x = anno[:19]
    anno_y = anno[19:]
    anno_x = [x if x >=0 else 0 for x in anno_x] 
    anno_x = [x if x <=image_width else image_width for x in anno_x] 
    anno_y = [y if y >=0 else 0 for y in anno_y] 
    anno_y = [y if y <=image_height else image_height for y in anno_y] 
    xmin, xmax, ymin, ymax = bbox
    
    xmin = max(xmin, 0)
    ymin = max(ymin, 0)
    xmax = min(xmax, image_width-1)
    ymax = min(ymax, image_height-1)

    image_crop = image[int(ymin):int(ymax)+1, int(xmin):int(xmax)+1, :]
    image_crop = cv2.resize(image_crop, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)

    anno_x = (np.array(anno_x) - xmin) / (xmax - xmin+1)
    anno_y = (np.array(anno_y) - ymin) / (ymax - ymin+1)

    anno = np.concatenate([anno_x.reshape(-1,1), anno_y.reshape(-1,1)], axis=1).flatten()
    anno = zip(anno[0::2], anno[1::2])
    return image_crop, list(anno)


def gen_data(root_folder, data_name, target_size, labeled_num_list):
    if not os.path.exists(os.path.join(root_folder, data_name)):
        os.mkdir(os.path.join(root_folder, data_name))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_train')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_train'))
    if not os.path.exists(os.path.join(root_folder, data_name, 'images_test')):
        os.mkdir(os.path.join(root_folder, data_name, 'images_test'))
    ############################################################################################
    if data_name == 'data_300W':
        # train for data_300W
        folders_train = ['afw', 'helen/trainset', 'lfpw/trainset']
        annos_train = {}
        for folder_train in folders_train:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_train)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, 'data_300W'), folder_train, image_name, label_name, target_size, 1.1)
                image_crop_name = folder_train.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
                annos_train[image_crop_name] = anno
        
        annos_train = list(annos_train.items())
        random.shuffle(annos_train)

        for labeled_num in labeled_num_list:
            assert labeled_num > 0 and labeled_num < len(annos_train)
            labeled_num_str = str(labeled_num)
            file_name_l = 'train_semi_l_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'
            file_name_u = 'train_semi_u_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'

            # labeled
            with open(os.path.join(root_folder, data_name, file_name_l), 'w') as f:
                for image_crop_name, anno in annos_train[:labeled_num]:
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')

            # unlabeled
            with open(os.path.join(root_folder, data_name, file_name_u), 'w') as f:
                for image_crop_name, anno in annos_train[labeled_num:]:
                    f.write(image_crop_name+'\n')

        # test for data_300W
        folders_test = ['helen/testset', 'lfpw/testset', 'ibug']
        annos_test = {}
        for folder_test in folders_test:
            all_files = sorted(os.listdir(os.path.join(root_folder, data_name, folder_test)))
            image_files = [x for x in all_files if '.pts' not in x]
            label_files = [x for x in all_files if '.pts' in x]
            assert len(image_files) == len(label_files)
            for image_name, label_name in zip(image_files, label_files):
                print(image_name)
                image_crop, anno = process_300w(os.path.join(root_folder, data_name), folder_test, image_name, label_name, target_size, 1.1)
                image_crop_name = folder_test.replace('/', '_')+'_'+image_name
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                annos_test[image_crop_name] = anno
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for image_crop_name, anno in annos_test.items():
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')

        annos = None
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'r') as f:
            annos = f.readlines()
    ############################################################################################
    elif data_name == 'WFLW':
        # WFLW train
        train_file = 'list_98pt_rect_attr_train.txt'
        with open(os.path.join(root_folder, data_name, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', train_file), 'r') as f:
            annos_train = f.readlines()
        annos_train = [x.strip().split() for x in annos_train]
        annos_train_dict = {}
        count = 1
        for anno_train in annos_train:
            image_crop, anno = process_wflw(anno_train, target_size)
            pad_num = 4-len(str(count))
            image_crop_name = 'wflw_train_' + '0' * pad_num + str(count) + '.jpg'
            print(image_crop_name)
            cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
            annos_train_dict[image_crop_name] = anno
            count += 1

        annos_train_dict = list(annos_train_dict.items())
        random.shuffle(annos_train_dict)

        for labeled_num in labeled_num_list:
            assert labeled_num > 0 and labeled_num < len(annos_train_dict)
            labeled_num_str = str(labeled_num)
            file_name_l = 'train_semi_l_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'
            file_name_u = 'train_semi_u_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'

            # labeled
            with open(os.path.join(root_folder, data_name, file_name_l), 'w') as f:
                for image_crop_name, anno in annos_train_dict[:labeled_num]:
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')

            # unlabeled
            with open(os.path.join(root_folder, data_name, file_name_u), 'w') as f:
                for image_crop_name, anno in annos_train_dict[labeled_num:]:
                    f.write(image_crop_name+'\n')

        # WFLW test
        test_file = 'list_98pt_rect_attr_test.txt'
        with open(os.path.join(root_folder, data_name, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', test_file), 'r') as f:
            annos_test = f.readlines()
        annos_test = [x.strip().split() for x in annos_test]
        names_mapping = {}
        count = 1
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for anno_test in annos_test:
                image_crop, anno = process_wflw(anno_test, target_size)
                pad_num = 4-len(str(count))
                image_crop_name = 'wflw_test_' + '0' * pad_num + str(count) + '.jpg'
                print(image_crop_name)
                names_mapping[anno_test[0]+'_'+anno_test[-1]] = [image_crop_name, anno]
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in list(anno):
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
                count += 1
    ############################################################################################
    elif data_name == 'AFLW':
        mat = hdf5storage.loadmat('../data/AFLW/AFLWinfo_release.mat')
        bboxes = mat['bbox']
        annos = mat['data']
        mask_new = mat['mask_new']
        nameList = mat['nameList']
        ra = mat['ra'][0]
        train_indices = ra[:20000]
        test_indices = ra[20000:]

        annos_train_dict = {}
        for index in train_indices:
            # from matlab index
            image_name = nameList[index-1][0][0]
            bbox = bboxes[index-1]
            anno = annos[index-1]
            image_crop, anno = process_aflw(root_folder, image_name, bbox, anno, target_size)
            pad_num = 5-len(str(index))
            image_crop_name = 'aflw_train_' + '0' * pad_num + str(index) + '.jpg'
            print(image_crop_name)
            cv2.imwrite(os.path.join(root_folder, data_name, 'images_train', image_crop_name), image_crop)
            annos_train_dict[image_crop_name] = anno

        annos_train_dict = list(annos_train_dict.items())
        random.shuffle(annos_train_dict)

        for labeled_num in labeled_num_list:
            assert labeled_num > 0 and labeled_num < len(annos_train_dict)
            labeled_num_str = str(labeled_num)
            file_name_l = 'train_semi_l_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'
            file_name_u = 'train_semi_u_'+'0'*(5-len(labeled_num_str))+labeled_num_str+'.txt'

            # labeled
            with open(os.path.join(root_folder, data_name, file_name_l), 'w') as f:
                for image_crop_name, anno in annos_train_dict[:labeled_num]:
                    f.write(image_crop_name+' ')
                    for x,y in anno:
                        f.write(str(x)+' '+str(y)+' ')
                    f.write('\n')

            # unlabeled
            with open(os.path.join(root_folder, data_name, file_name_u), 'w') as f:
                for image_crop_name, anno in annos_train_dict[labeled_num:]:
                    f.write(image_crop_name+'\n')

        # test for AFLW
        with open(os.path.join(root_folder, data_name, 'test.txt'), 'w') as f:
            for index in test_indices:
                # from matlab index
                image_name = nameList[index-1][0][0]
                bbox = bboxes[index-1]
                anno = annos[index-1]
                image_crop, anno = process_aflw(root_folder, image_name, bbox, anno, target_size)
                pad_num = 5-len(str(index))
                image_crop_name = 'aflw_test_' + '0' * pad_num + str(index) + '.jpg'
                print(image_crop_name)
                cv2.imwrite(os.path.join(root_folder, data_name, 'images_test', image_crop_name), image_crop)
                f.write(image_crop_name+' ')
                for x,y in anno:
                    f.write(str(x)+' '+str(y)+' ')
                f.write('\n')
    else:
        print('Wrong data!')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('please input one of the data names:')
        print('data_300W')
        print('WFLW')
        print('AFLW')
        exit(0)
    else:
        data_name = sys.argv[1]
        labeled_num_list = {}
        labeled_num_list['data_300W'] = [630, 315, 157, 50]
        labeled_num_list['WFLW'] = [1500, 750, 375, 50]
        labeled_num_list['AFLW'] = [4000, 2000, 1000, 200]
        assert data_name in labeled_num_list
        gen_data('../data', data_name, (256, 256), labeled_num_list[data_name])
