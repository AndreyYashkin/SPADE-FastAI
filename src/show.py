from fastai.vision.all import *

def plot_record_fn(path, batch, output):
    nrows=None
    ncols=None
    figsize=None
    n = output.shape[0]
    fig, ctxs = get_grid(n, nrows=nrows, ncols=ncols, add_vert=1, figsize=figsize, double=True, return_fig=True)
    seg = torch.argmax(batch[1], dim=1)
    seg = torch.split(seg, 1)
    ctxs[0::2] = [b.show(ctx=c) for b,c,_ in zip(seg,ctxs[0::2],range(n))]
    output = torch.clamp(output*0.5 + 0.5, 0, 1) # denorm
    ims = seg = torch.split(output, 1)
    ctxs[1::2] = [b[0].show(ctx=c) for b,c,_ in zip(ims,ctxs[1::2],range(n))]
    fig.subplots_adjust(left=0.0,
                        bottom=0.0,
                        right=1.0,
                        top=1.0)
    fig.savefig(path)
    plt.close()
