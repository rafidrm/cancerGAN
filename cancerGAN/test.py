import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html
import pudb

opt = TestOptions().parse()
opt.nThreads = 1  # test code only supports 1 thread
opt.batchSize = 1  # test code only supports 1 batch
opt.serial_batches = True
opt.no_flip = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(
    opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = {}, Phase = {}, Epoch = {}'.format(
    opt.name, opt.phase, opt.which_epoch))

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%05d: process image... %s' % (i, img_path))
    visualizer.save_images(
        webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

webpage.save()
