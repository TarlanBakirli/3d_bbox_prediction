import torch

class ToTensor(object):

    def __call__(self, sample):
        image, target, pc, mask = sample['image'], sample['target'], sample['pc'], sample['mask']

        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        bbox = torch.from_numpy(target['bbox'])
        target['bbox'] = bbox
        pc = torch.from_numpy(pc)
        mask = torch.from_numpy(mask)

        return {'target': target, 'image': image, 'pc': pc, 'mask': mask}