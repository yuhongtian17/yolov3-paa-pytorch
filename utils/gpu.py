# coding=utf-8

import torch


def select_device(id, force_cpu = False):
    cuda = False if force_cpu else torch.cuda.is_available()
    device = torch.device("cuda:{}".format(id) if cuda else "cpu")

    #if cuda:
    #    c = 1024 ** 2  # bytes to MB
    #    ng = torch.cuda.device_count()
    #    x = [torch.cuda.get_device_properties(i) for i in range(ng)]
    #    if ng >= 0:
    #        for i in range(ng):
    #            print("Using CUDA device{} _CudaDeviceProperties(name = "{}", total_memory = {} MB)".format(i, x[i].name, x[i].total_memory / c))
    #else:
    #    print("Using CPU")

    return device



if __name__ == "__main__":
    _ = select_device(0)

