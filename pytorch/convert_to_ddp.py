import argparse
import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def get_args():
    parser = argparse.ArgumentParser("training for knowledge distillation.")
    parser.add_argument('--local_rank', type=int, help='local rank, will passed by ddp')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument("--world-size", default=4, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", type=str, help="url used to set up distributed training")

    args = parser.parse_args()
    return args


def init_distributed_mode(args):
    '''initilize DDP
    '''
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    elif hasattr(args, "rank"):
        pass
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print(
        f"| distributed init (rank {args.rank}): {args.dist_url}, local rank:{args.gpu}, world size:{args.world_size}",
        flush=True)
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
    )


def convert_to_ddp(model, train_dataset, test_dataset, batch_size):
    args = get_args()
    init_distributed_mode(args)

    model = model.cuda()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = DDP(model, device_ids=[args.gpu])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

    # 需要注意的是，这里的batch_size指的是每个进程下的batch_size。也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=False, num_workers=4, pin_memory=True,
                                               sampler=train_sampler)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=True)

    return model, train_loader, test_loader
