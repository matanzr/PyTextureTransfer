__author__ = 'matanzohar'
import sys
import random
import numpy as np
from scipy import misc, ndimage

source_path = 'gogh2_h.jpg'
target_path = 'earth_h.jpg'

class ExhastiveProbe(object):
    def __init__(self, width, height):
        self.counter = 0
        self.width = width
        self.height = height

    def next(self):
        Y = self.counter / self.width
        X = self.counter - self.width * Y
        if (self.counter >= self.width * self.height):
            probeCompleted = True
        else:
            probeCompleted = False
        self.counter += 1
        return (X, Y, probeCompleted)

    def all_locations(self):
        l = np.arange(0, self.width,self.height, 1)
        y = l / self.width
        x = l - self.width * y
        return np.dstack((x,y))

    def reset(self):
        self.counter = 0

    def respawn(self, width, height):
        self.reset()
        self.width = width
        self.height = height


class RandomProbe(object):
    def __init__(self, width, height, fractionToProbe):
        self.counter = 0
        self.width = width
        self.height = height
        self.max_attempts = int(fractionToProbe * width * height)

    def next(self):
        i = random.randint(0, self.width * self.height)
        Y = i / self.width
        X = i - self.width * Y
        if (self.counter >= self.max_attempts):
            probeCompleted = True
        else:
            probeCompleted = False
        self.counter += 1
        return (X, Y, probeCompleted)

    def all_locations(self):
        r = np.random.randint(0, self.width * self.height, self.max_attempts)
        y = r / self.width
        x = r - self.width * y
        return np.dstack((x,y))[0]

    def reset(self):
        self.counter = 0

    def respawn(self, width, height):
        self.reset()
        self.width = width
        self.height = height

def get_block(image, topLeft, width, height):
    return image[topLeft[1]:topLeft[1] + height, topLeft[0]:topLeft[0] + width].copy()

def block_diff(bl1, bl2):
    d = bl1.astype(np.float64) - bl2
    return np.sum(d * d)

def compare_blocks(img_block, target_block, src, locations, alpha, isFirstIteration, overlapX, overlapY, blockWidth, blockHeight):
    samples = np.empty((len(locations), blockHeight, blockWidth, 3))
    for i in range(len(locations)): samples[i] = get_block(src, locations[i], blockWidth, blockHeight)
    d1 = samples[:] - img_block
    if isFirstIteration:
        d1[:,overlapY:-1, overlapX:-1, :] = [0, 0, 0]
    d1 = np.sum(d1 * d1, axis = (1,2,3))

    d2 = samples - target_block
    d2 = np.sum(d2 * d2, axis= (1,2,3))
    return samples, alpha * d1 + (1.0 - alpha) * d2

def get_neighbors(locations, size, max_w, max_h):
    neighbors = None
    for i in locations:
        x = np.arange(max(0,i[0] - size), min(i[0] + size, max_w) , 1)
        y = np.arange(max(0,i[1] - size), min(i[1] + size, max_h), 1)
        if neighbors is None:
            neighbors = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
        else:
            neighbors = np.concatenate((neighbors,np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])), axis = 0)
    return neighbors

def get_matching_block(img, # image being synthesized
                       src, # source image
                       target, # target image
                       topLeft, # topLeft corner of block to be synthesized in img
                       overlapX,
                       overlapY,
                       blockWidth,
                       blockHeight,
                       alpha,
                       isFirstIteration,
                       probe):
    p = (0, 0)
    srcWidth, srcHeight = img.shape[:2]
    dmin = sys.float_info.max
    probe_completed = False
    probe.reset()
    while(not probe_completed):
        x, y, probe_completed = probe.next()
        d = d1 = d2 = 0.0
        new_block = get_block(src, (x,y), blockWidth, blockHeight)
        d1 = get_block(img, topLeft, blockWidth, blockHeight) - new_block
        diff1 = np.sum(d1*d1, 2)
        #diff1 = block_diff(get_block(img, topLeft, blockWidth, blockHeight), new_block)
        if isFirstIteration:
            d1 = np.sum(diff1[0: overlapY]) + np.sum(diff1[overlapY: -1, 0: overlapX])
        else:
            d1 = np.sum(diff1)
        d2 = block_diff(get_block(target, topLeft, blockWidth, blockHeight),new_block)

        #d2 = np.sum(np.power(d2[..., 0],2) + np.power(d2[..., 1],2) + np.power(d2[..., 2], 2))
        d = alpha * d1 + (1.0 - alpha) * d2
        if d < dmin:
            dmin = d
            p = (x, y)
            cut = diff1

    return p, cut[0: overlapY], cut[..., 0: overlapX]

def cut_and_paste(src, src_top_left, dest, dest_top_left, h_cut, v_cut, block_width, block_height, display_boundary_cut):
    cut_mask = np.zeros((block_height, block_width))
    if not h_cut is None:
        for i in range(len(h_cut)):
            cut_mask[0:h_cut[i], i] = 1
    if not v_cut is None:
        for i in range(len(v_cut)):
            cut_mask[i, 0: v_cut[i]] = 1
    s = src[ src_top_left[1]: src_top_left[1] + block_height, src_top_left[0] : src_top_left[0] + block_width].copy()
    d = dest[ dest_top_left[1]: dest_top_left[1] + block_height, dest_top_left[0] : dest_top_left[0] + block_width]
    cut_mask = cut_mask
    s[cut_mask == 1] = d[cut_mask == 1]
    dest[ dest_top_left[1]: dest_top_left[1] + block_height, dest_top_left[0] : dest_top_left[0] + block_width] = s


def find_h_cut(cut):
    overlap_y = cut.shape[0]
    if overlap_y < 3: return None # overlap too small
    w1width = cut.shape[1]

    min_vals = np.empty_like(cut)
    min_vals[:, 0] = cut[:, 0]

    for i in range(1, w1width, 1):
        min_vals[:, i] = cut[:, i] + ndimage.minimum_filter(min_vals[:,i-1:i+1], footprint = [[1,0], [1,0], [1,0]], mode = 'constant')[:,1]
        min_vals[0, i] = cut[0, i] + min (min_vals[0, i-1], min_vals[1, i-1])
        min_vals[-1, i] = cut[-1, i] + min (min_vals[-1, i-1], min_vals[-1, i-1])

    res = np.empty(w1width)
    res[-1] = np.argmin(min_vals[:,-1])
    for i in range(w1width-2, -1, -1):
        res[i] = res[i+1] + np.argmin(min_vals[max(0,res[i+1]-1): min(overlap_y,res[i+1]+2), i]) - 1 + int(res[i+1]-1 < 0)
    return res


def find_v_cut(cut):
    overlap_x = cut.shape[1]
    if overlap_x < 3: return None
    w1height = cut.shape[0]

    min_vals = np.empty_like(cut)
    min_vals[0] = cut[0]

    for i in range(1, w1height, 1):
        min_vals[i] = cut[i] + ndimage.minimum_filter(min_vals[i-1:i+1], footprint = [[1,1,1], [0, 0, 0]], mode = 'constant')[1]
        min_vals[i, 0] = cut[i, 0] + min (min_vals[i-1, 0], min_vals[i-1, 1])
        min_vals[i, -1] = cut[i, -1] + min (min_vals[i-1, -1], min_vals[-i-1, -1])

    res = np.empty(w1height)
    res[-1] = np.argmin(min_vals[-1])
    for i in range(w1height-2, -1, -1):
        res[i] = res[i+1] + np.argmin(min_vals[i, max(res[i+1]-1, 0): min(overlap_x,res[i+1]+2)]) - 1 + int(res[i+1]-1 < 0)

    return res


class TextureTransferTool(object):
    def __init__(self, source, target, blockW, blockH, num_of_iterations, overlap_x_frac, overlap_y_frac,
                 block_reduciton_factor, amount_to_probe_frac, display_boundary_cut):
        self.source =  misc.imread(source)
        self.target =  misc.imread(target)
        self.block_width = blockW
        self.block_height = blockH
        self.num_of_iterations= num_of_iterations
        self.overlap_x_fraction = overlap_x_frac
        self.overlap_y_fraction = overlap_y_frac
        self.block_reduction_factor = block_reduciton_factor
        self.amount_to_probe_fraction = amount_to_probe_frac
        self.display_boundary_cut = display_boundary_cut

    def next(self, x, y):
        completed = False
        image_width = self.target.shape[1]
        overlap_x = int(round(self.overlap_x_fraction * self.block_width))
        overlap_y = int(round(self.overlap_y_fraction * self.block_height))
        next_x = x + self.block_width - overlap_x
        if next_x >= image_width - overlap_x:
            next_x = 0
            next_y = y + self.block_height - overlap_y
            if (next_y >= self.target.shape[0] - overlap_y): completed = True
        else:
            next_y = y

        next_block_width = min(self.block_width, image_width - next_x)
        next_block_height = min(self.block_height, self.target.shape[0] - next_y)
        next_overlap_x = overlap_x * int(not next_x == 0)
        next_overlap_y = overlap_y * int(not next_y == 0)
        return next_x, next_y, next_block_width, next_block_height, next_overlap_x, next_overlap_y, completed


    def start(self, outfile):
        img_width = self.target.shape[1]
        img_height = self.target.shape[0]
        src_width = self.source.shape[1]
        src_height = self.source.shape[0]
        img = np.zeros((img_height,img_width,3))
        probe_width = src_width - self.block_width
        probe_height = src_height - self.block_height
        if self.amount_to_probe_fraction >= 1:
            probe = ExhastiveProbe(probe_width, probe_height)
        else:
            probe = RandomProbe(probe_width, probe_height, self.amount_to_probe_fraction)

        for i in range(self.num_of_iterations):
            is_first_iteration = i == 0
            alpha = 0.8 * i / (self.num_of_iterations - 1) + 0.1  # from Alyosha's paper
            next_x = next_y = 0; next_block_width = self.block_width; next_block_height = self.block_height
            next_overlap_x = next_overlap_y = 0
            completed = False

            while(not completed):
                q = (next_x, next_y)
                # get matching block from src which will be pasted at location q in img
                p, eh, ev = get_matching_block2(img, self.source, self.target, q, next_overlap_x, next_overlap_y,
                                       next_block_width, next_block_height, alpha, is_first_iteration, probe)
                 # the horizontal and vertical cuts will ensure that the patch from src will fit in
                 # seamlessly when it is pasted onto img
                h_cut = find_h_cut(eh);  # compute the horizontal cut
                v_cut = find_v_cut(ev);  # compute the vertical cut
                cut_and_paste(self.source, p, img, q, h_cut, v_cut, next_block_width, next_block_height, self.display_boundary_cut)
                next_x, next_y, next_block_width, next_block_height, next_overlap_x, next_overlap_y, completed = self.next(next_x, next_y)
                #if next_x % 500 ==0 :
                #    misc.imsave(outfile, img)
            self.block_width = int(round(self.block_width * self.block_reduction_factor))
            self.block_height = int(round(self.block_height * self.block_reduction_factor))
            probeWidth = self.source.shape[1] - self.block_width
            probeHeight = self.source.shape[0] - self.block_height
            probe.respawn(probeWidth, probeHeight)
            if i > 0: misc.imsave(outfile+str(i)+".png", img)
