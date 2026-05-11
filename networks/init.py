import torch
from networks.GEL_cache import TransformerModel


def init_model(args, input_dim, input_sdim):
    if args.module != "GEL":
        raise ValueError(f"Unsupported module: {args.module}")

    model = TransformerModel(
        args.hops,
        args.n_hidden,
        input_dim + args.pe_dim,
        args.pe_dim,
        n_layers=args.n_layers,
        hidden_dim=args.n_hidden * 2,
        dropout_rate=args.dropout,
    )

    if args.gpu >= 0:
        device = torch.device(f"cuda:{args.gpu}")
        model.to(device)

    return model