import os
import pudb
import shutil
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from scipy.io import loadmat, savemat


def ttsplit_and_copy(aaron_dir, data_dir, train, test):
    img_list = os.listdir(aaron_dir)
    x_train, x_test = train_test_split(img_list, train_size=train)
    x_test, x_val = train_test_split(x_test, train_size=test)
    for fname in tqdm(x_train):
        shutil.copy(os.path.join(aaron_dir, fname),
                    os.path.join(data_dir, 'train', fname))
    for fname in tqdm(x_test):
        shutil.copy(os.path.join(aaron_dir, fname),
                    os.path.join(data_dir, 'test', fname))
    for fname in tqdm(x_val):
        shutil.copy(os.path.join(aaron_dir, fname),
                    os.path.join(data_dir, 'val', fname))


def move_to_cancerGAN(aaron_dir, data_dir, new_dir=None, train=0.6, test=0.5):
    ''' Taking aaron's jpegs and parsed them. '''
    img_list = os.listdir(aaron_dir)
    num_files = len(img_list)
    if new_dir is not None:
        for i in tqdm(range(num_files)):
            shutil.copy(os.path.join(aaron_dir, img_list[i]),
                        os.path.join(new_dir, '{}.jpg'.format(i + 1)))
        aaron_dir = new_dir

    ttsplit_and_copy(aaron_dir, data_dir, train, test)


def collect_parse_mat_slices(aaron_dir, data_dir, new_dir=None, train=0.6, test=0.5):
    ''' Takes 2 folders, merges them appropriately, then ttsplit and resave.'''
    if new_dir is not None and len(aaron_dir) == 2:
        clin_dir == aaron_dir[0]
        ct_dir = aaron_dir[1]
        clin_list = os.listdir(clin_dir)
        ct_list = os.listdir(ct_dir)
        for clinFile in tqdm(clin_list):
            if os.path.isfile(os.path.join(ct_dir, clinFile)):
                clin = loadmat(os.path.join(clin_dir, clinFile))
                ct = loadmat(os.path.join(ct_dir, clinFile))
                mDict = {'dMs': clin['dMs'], 'iMs': ct['iMs']}
                saveFile = os.path.join(new_dir, clinFile)
                savemat(saveFile, mDict)
            else:
                print(
                    'File [{}] does not exist in CT directory'.format(clinFile))
        aaron_dir = new_dir
    elif len(aaron_dir > 1):
        raise ValueError

    ttsplit_and_copy(aaron_dir, data_dir, train, test)


if __name__ == '__main__':
    # aaron_dir = 'slices-truncated/'
    # new_dir = 'slices/'
    # data_dir = '/home/rm/Python/cancerGAN/cancerGAN/datasets/cancer'
    # move_to_cancerGAN(aaron_dir, data_dir)

    clin_dir = 'Clin_dose_Mat_files/'
    ct_dir = 'CT_Mat_files/'
    aaron_dir = (clin_dir, ct_dir)
    data_dir = '/home/rm/Python/cancerGAN/cancerGAN/datasets/cancer_mat'
    collect_parse_mat_slices(aaron_dir, data_dir, new_dir='merged_Mat')
