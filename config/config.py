# from easydict import EasyDict as edict

# config = edict()
def lr_step_func(epoch):
    return (
        ((epoch + 1) / (4 + 1)) ** 2
        if epoch < -1
        else 0.1 ** len([m for m in [8, 14, 20, 25] if m - 1 <= epoch])
    )  # [m for m in [8, 14,20,25] if m - 1 <= epoch])

config = dict()

config["la"] = 10
config["ua"] = 110
config["l_margin"] = 0.45
config["u_margin"] = 0.8
config["lr"] = 0.1
config["weight_decay"] = 5e-4
config["s"] = 64.0
config["m"] = 0.50
config["lr_func"] = lr_step_func
