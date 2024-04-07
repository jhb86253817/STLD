
class Config():
    def __init__(self):
        self.labeled_num = 630
        self.os_num = 1000
        self.semi_iter = 5
        self.shrink_curri = [1.5, 2.2, 1.8, 1.5, 1.5]
        self.shrink_weight_curri = [1, 0.1, 0.1, 1, 1]
        self.det_head = 'map'
        self.net_stride = 4
        self.input_size = (256, 256)
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 60
        self.decay_steps = [40, 50]
        self.backbone = 'hrnet'
        self.gt_sigma = 1.5
        self.num_lms = 68
        self.use_gpu = True
        self.gpu_id = 0
