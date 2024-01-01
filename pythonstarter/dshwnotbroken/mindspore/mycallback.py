from mindspore.train.callback import Callback

# 定义一个Callback，用于在每个epoch结束时执行操作
class MyCallback(Callback):
    def __init__(self):
        super(MyCallback, self).__init__()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()
        epoch = cb_params.cur_epoch_num
        loss = cb_params.net_outputs
        print(f"Epoch {epoch}, Loss: {loss}")

# custom callback function
class StepLossAccInfo(Callback):
    def __init__(self, model, eval_dataset, steps_loss, steps_eval):
        self.model = model
        self.eval_dataset = eval_dataset
        self.steps_loss = steps_loss
        self.steps_eval = steps_eval

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        cur_step = (cur_epoch-1)*1304 + cb_params.cur_step_num
        self.steps_loss["loss_value"].append(str(cb_params.net_outputs))
        self.steps_loss["step"].append(str(cur_step))
        if cur_step % 125 == 0:
            acc = self.model.eval(self.eval_dataset, dataset_sink_mode=False)
            self.steps_eval["step"].append(cur_step)
            self.steps_eval["acc"].append(acc["Accuracy"])

