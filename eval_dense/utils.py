import urllib
import os
import os.path as osp
from torch.utils.tensorboard import SummaryWriter


def prepare_dirs(exp_type, args):
    os.umask(0x0002)

    setup_name = "{}/{}_{}_B{}_I{}_lr{}_{}".format(
        exp_type,
        args.dataset,
        args.head_type,
        args.samples_per_gpu,
        args.iter_with_8_gpu,
        args.lr,
        args.seed,
    )

    if args.output_dir == "":
        # automatically set output dir
        args.output_dir = osp.join(
            osp.dirname(args.pretrained), setup_name.replace("/", "_")
        )
        print("output_dir is set to {}".format(args.output_dir))

    if args.purge_output_dir and osp.isdir(args.output_dir):
        print("Purging the output dir ({})".format(args.output_dir))
        os.system(f"rm -rf {args.output_dir}")

    return setup_name


def read_final_log_file(output_dir):
    # parse the json to get the final results
    json_files = sorted([f for f in os.listdir(output_dir) if f.endswith("log.json")])
    assert len(json_files) > 0
    with open(osp.join(output_dir, json_files[-1]), "r") as fid:
        lines = fid.readlines()

    return lines


def check_for_results(args, res_fname="results.txt"):
    res_fname = osp.join(args.output_dir, res_fname)

    result = None
    if osp.isfile(res_fname):
        with open(res_fname, "r") as fid:
            result = float(fid.read())

    return res_fname, result


def get_backbone_name_from_model(model):
    if model.embed_dim == 384 and model.n_blocks == 12:
        return "dinov2_vits14"
    elif model.embed_dim == 768 and model.n_blocks == 12:
        return "dinov2_vitb14"
    elif model.embed_dim == 1024 and model.n_blocks == 24:
        return "dinov2_vitl14"
    elif model.embed_dim == 1536 and model.n_blocks == 40:
        return "dinov2_vitg14"
    else:
        raise ValueError("Cannot find model size")


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


def log_tensorboard(setupname, to_log_dict, step, pretrained_path):
    tbdir = osp.join(osp.dirname(pretrained_path), "tb")
    if not os.path.isdir(tbdir):
        print("Skipping tensorboard logging as the tb/ dir does not exist: " + tbdir)
        return
    writer = SummaryWriter(log_dir=tbdir, filename_suffix=setupname.replace("/", "_"))
    for k, v in to_log_dict.items():
        writer.add_scalar(setupname + "_" + k, v, step)
    writer.close()


def delete_checkpoints(output_dir):
    print("deleting checkpoints from output_dir: ", output_dir)
    for f in os.listdir(output_dir):
        if f.endswith(".pth"):
            if os.path.islink(f):  # latest.pth
                os.system(f'unlink "{osp.join(output_dir,f)}"')
            else:  # other ckpts
                os.remove(osp.join(output_dir, f))
