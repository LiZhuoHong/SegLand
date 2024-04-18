from loss.criterion import CELoss, OrthLoss

def get_loss(args):
    if 'pop' in args.model:
        criterion = OrthLoss(ignore_index=args.ignore_label)
    else:
        criterion = CELoss(ignore_index=args.ignore_label)
    return criterion