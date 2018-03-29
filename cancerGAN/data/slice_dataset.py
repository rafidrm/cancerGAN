import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from scipy.io import loadmat
import pudb


class SliceDataset(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # go through directory return os.path for all images
        slice_filetype = ['.mat']
        self.AB_paths = sorted(make_dataset(self.dir_AB, slice_filetype))
        assert self.opt.loadSize == self.opt.fineSize, 'No resize or cropping.'
        assert self.which_direction == 'AtoB', 'enforce A to B for slices.'

    def __getitem__(self, index):
        AB_path = self.AB_paths[index]

        mat = loadmat(AB_path)
        dose_img = mat['dMs']
        ct_img = mat['iMs']
        w, h, nc = ct_img.shape
        assert w == self.opt.loadSize, 'size mismatch in width'
        assert h == self.opt.loadSize, 'size mismatch in height'

        A = transforms.ToTensor()(ct_img).float()
        B = transforms.ToTensor()(dose_img).float()

        # ABs are 3-channel. Normalizing to 0.5 mean, 0.5 std
        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        # flipping augments the dataset by flipping a bunch of the images.
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        # by default assume 3 channels, but if 1 channel then have to greyscale
        if input_nc == 1:   # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:   # RGB to gray
            tmp = B[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.AB_paths)

    def name(self):
        return 'SliceDataset'
