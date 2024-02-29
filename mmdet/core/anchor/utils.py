import torch


def images_to_levels(target, num_levels):
    """Convert targets by image to targets by feature level.

    [target_img0, target_img1] -> [target_level0, target_level1, ...]
    """
    #print(type(target[0]))
    target = torch.stack(target,0)



    level_targets = []
    start = 0
    # torch.nn.functional.interpolate(input, size=None, scale_factor=None, mode='nearest', align_corners=None)
#     transforms.Resize((img_size, img_size))
# def collate_fn(batch):
#     batch = list(filter(lambda x: x is not None, batch))
#     return torch.utils.data.dataloader.default_collate(batch) 
# def collate_fn(self,data):
#     imgs_list,boxes_list,classes_list=zip(*data)
#     assert len(imgs_list)==len(boxes_list)==len(classes_list)
#     batch_size=len(boxes_list)
#     pad_imgs_list=[]
#     pad_boxes_list=[]
#     pad_classes_list=[]
 
#     h_list = [int(s.shape[1]) for s in imgs_list]
#     w_list = [int(s.shape[2]) for s in imgs_list]
#     max_h = np.array(h_list).max()
#     max_w = np.array(w_list).max()
#     for i in range(batch_size):
#         img=imgs_list[i]
#         pad_imgs_list.append(torch.nn.functional.pad(img,(0,int(max_w-img.shape[2]),0,int(max_h-img.shape[1])),value=0.))
 
#     max_num=0
#     for i in range(batch_size):
#         n=boxes_list[i].shape[0]
#         if n>max_num:max_num=n
#     for i in range(batch_size):
#         pad_boxes_list.append(torch.nn.functional.pad(boxes_list[i],(0,0,0,max_num-boxes_list[i].shape[0]),value=-1))
#         pad_classes_list.append(torch.nn.functional.pad(classes_list[i],(0,max_num-classes_list[i].shape[0]),value=-1))
 
 
#     batch_boxes=torch.stack(pad_boxes_list)
#     batch_classes=torch.stack(pad_classes_list)
#     batch_imgs=torch.stack(pad_imgs_list)
 
#     return batch_imgs,batch_boxes,batch_classes
    #
    for n in num_levels:
        end = start + n
        # level_targets.append(target[:, start:end].squeeze(0))
        level_targets.append(target[:, start:end])
        start = end
    return level_targets


def anchor_inside_flags(flat_anchors,
                        valid_flags,
                        img_shape,
                        allowed_border=0):
    """Check whether the anchors are inside the border.

    Args:
        flat_anchors (torch.Tensor): Flatten anchors, shape (n, 4).
        valid_flags (torch.Tensor): An existing valid flags of anchors.
        img_shape (tuple(int)): Shape of current image.
        allowed_border (int, optional): The border to allow the valid anchor.
            Defaults to 0.

    Returns:
        torch.Tensor: Flags indicating whether the anchors are inside a \
            valid range.
    """
    img_h, img_w = img_shape[:2]
    if allowed_border >= 0:
        inside_flags = valid_flags & \
            (flat_anchors[:, 0] >= -allowed_border) & \
            (flat_anchors[:, 1] >= -allowed_border) & \
            (flat_anchors[:, 2] < img_w + allowed_border) & \
            (flat_anchors[:, 3] < img_h + allowed_border)
    else:
        inside_flags = valid_flags
    return inside_flags


def calc_region(bbox, ratio, featmap_size=None):
    """Calculate a proportional bbox region.

    The bbox center are fixed and the new h' and w' is h * ratio and w * ratio.

    Args:
        bbox (Tensor): Bboxes to calculate regions, shape (n, 4).
        ratio (float): Ratio of the output region.
        featmap_size (tuple): Feature map size used for clipping the boundary.

    Returns:
        tuple: x1, y1, x2, y2
    """
    x1 = torch.round((1 - ratio) * bbox[0] + ratio * bbox[2]).long()
    y1 = torch.round((1 - ratio) * bbox[1] + ratio * bbox[3]).long()
    x2 = torch.round(ratio * bbox[0] + (1 - ratio) * bbox[2]).long()
    y2 = torch.round(ratio * bbox[1] + (1 - ratio) * bbox[3]).long()
    if featmap_size is not None:
        x1 = x1.clamp(min=0, max=featmap_size[1])
        y1 = y1.clamp(min=0, max=featmap_size[0])
        x2 = x2.clamp(min=0, max=featmap_size[1])
        y2 = y2.clamp(min=0, max=featmap_size[0])
    return (x1, y1, x2, y2)
