
class Config():
    def __init__(self):
        self.labeled_num = 750
        self.semi_iter = 5
        self.shrink_curri = [1, 2.4, 1.6, 1, 1]
        self.shrink_weight_curri = [1, 0.1, 0.1, 1, 1]
        self.det_head = 'tf'
        self.net_stride = 32
        self.input_size = (256, 256)
        self.batch_size = 16
        self.init_lr = 0.0001
        self.num_epochs = 360
        self.decay_steps = [240, 320]
        self.backbone = 'resnet18'
        self.gt_sigma = None
        self.num_lms = 98
        self.use_gpu = True
        self.gpu_id = 2
        ##############
        self.tf_dim = 256
        self.tf_en_num = 0
        self.tf_de_num = 4
        self.dynamic_query = True
        self.dq_pos = True
        self.qa_attn = True 
        self.qa_group = 1
        self.sigmoid = False 
        ##############
