import numpy as np

# generate artificial data
nr_of_bb = 4000

# minimal and maximal position and dimension of rectangles
min_xywh = [.0,.0,.2,.2]
max_xywh = [1.,1.,1.,1.]

# four lists of rectangles:\
# - bbs1 and bbs2 are used to generate examples R(x,y) with x in bbs1 and y in bbs2;
# - bbs12 = bbs1 + bbs2
# - bbst is the set of rectangles for test
bbs1 = np.random.uniform(min_xywh,max_xywh, size=(nr_of_bb, 4))
bbs2 = np.random.uniform(min_xywh,max_xywh, size=(nr_of_bb, 4))
bbs12 = np.concatenate([bbs1,bbs2],axis=0)
bbst = np.random.uniform([0, 0, .2, .2], [1, 1, 1, 1], size=(nr_of_bb, 4))

# funcitions that ocmputes training examples or relations between BB

def angle(bb1,bb2):
    c1 = bb1[:2] + .5*bb1[2:]
    c2 = bb2[:2] + .5*bb2[2:]
    x = c2 - c1
    return np.angle(x[0] + 1j*x[1],deg=True)

def is_left(bb1,bb2):
    return bb1[0] + bb1[2] < bb2[0] and np.abs(angle(bb1, bb2)) < 5

def is_not_left(bb1,bb2):
    return bb1[0] + bb1[2] > bb2[0] or np.abs(angle(bb1, bb2)) > 45

def is_right(bb1, bb2):
    return is_left(bb2,bb1)

def is_not_right(bb1,bb2):
    return is_not_left(bb2,bb1)

def is_below(bb1, bb2):
    return bb1[1] + bb1[3] < bb2[1] and np.abs(angle(bb1, bb2)-90) < 5

def is_not_below(bb1, bb2):
    return bb1[1] + bb1[3] > bb2[1] or np.abs(angle(bb1, bb2)-90) > 45

def is_above(bb1, bb2):
    return is_below(bb2,bb1)

def is_not_above(bb1,bb2):
    return is_not_below(bb2,bb1)

def contains(bb1,bb2):
    return bb1[0] < bb2[0] and bb1[0] + bb1[2] > bb2[0] + bb2[2] and \
           bb1[1] < bb2[1] and bb1[1] + bb1[3] > bb2[1] + bb2[3]

def not_contains(bb1,bb2):
    return not contains(bb1,bb2)

def is_in(bb1,bb2):
    return contains(bb2,bb1)

def is_not_in(bb1,bb2):
    return not is_in(bb1,bb2)

# pairs of rectangles for training

left_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_left(bbs1[i],bbs2[i])])
left_data = np.squeeze(left_data)

right_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_right(bbs1[i],bbs2[i])])
right_data = np.squeeze(right_data)

above_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_above(bbs1[i],bbs2[i])])
above_data = np.squeeze(above_data)

below_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_below(bbs1[i],bbs2[i])])
below_data = np.squeeze(below_data)

contain_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if contains(bbs1[i],bbs2[i])])
contain_data = np.squeeze(contain_data)

in_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_in(bbs1[i],bbs2[i])])
in_data = np.squeeze(in_data)

non_left_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_not_left(bbs1[i],bbs2[i])])
non_left_data = np.squeeze(non_left_data)

non_right_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_not_right(bbs1[i],bbs2[i])])
not_right_data = np.squeeze(non_right_data)

non_above_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_not_above(bbs1[i],bbs2[i])])
non_above_data = np.squeeze(non_above_data)

non_below_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_not_below(bbs1[i],bbs2[i])])
non_below_data = np.squeeze(non_below_data)

non_contain_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if not_contains(bbs1[i],bbs2[i])])
non_contain_data = np.squeeze(non_contain_data)

non_in_data = np.array([np.concatenate([bbs1[i],bbs2[i]],axis=0)
             for i in range(nr_of_bb)
             if is_not_in(bbs1[i],bbs2[i])])
non_in_data = np.squeeze(non_in_data)

def generate_rectangles(nr_rectangles,min_xywh = [.0,.0,.2,.2],max_xywh = [1.,1.,1.,1.]):
    return np.random.uniform(min_xywh,max_xywh, size=(nr_rectangles, 4)).astype("float32")

def angle(bbs1,bbs2):
    c1 = bbs1[:,:2] + .5*bbs1[:,2:]
    c2 = bbs2[:,:2] + .5*bbs2[:,2:]
    x = c2 - c1
    return np.angle(x[:,0] + 1j*x[:,1],deg=True)

def is_left(bbs1,bbs2):
    return np.logical_and(bbs1[:,0] + bbs1[:,2] < bbs2[:,0],np.abs(angle(bbs1, bbs2)) < 5.)

def is_not_left(bbs1,bbs2):
    return np.logical_or(bbs1[:,0] + bbs1[:,2] > bbs2[:,0], np.abs(angle(bbs1, bbs2)) > 45)

def is_right(bbs1, bbs2):
    return is_left(bbs2,bbs1)

def is_not_right(bbs1,bbs2):
    return is_not_left(bbs2,bbs1)

def is_below(bbs1, bbs2):
    return np.logical_and(bbs1[:,1] + bbs1[:,3] < bbs2[:,1],np.abs(angle(bbs1, bbs2)-90) < 5)

def is_not_below(bbs1, bbs2):
    return np.logical_or(bbs1[:,1] + bbs1[:,3] > bbs2[:,1],np.abs(angle(bbs1, bbs2)-90) > 45)

def is_above(bbs1, bbs2):
    return is_below(bbs2,bbs1)

def is_not_above(bbs1,bbs2):
    return is_not_below(bbs2,bbs1)

def contains(bbs1,bbs2):
    return np.all([bbs1[:,0] < bbs2[:,0], 
                          bbs1[:,0] + bbs1[:,2] > bbs2[:,0] + bbs2[:,2],
                          bbs1[:,1] < bbs2[:,1],
                          bbs1[:,1] + bbs1[:,3] > bbs2[:,1] + bbs2[:,3]],axis=0)

def not_contains(bbs1,bbs2):
    return np.logical_not(contains(bbs1,bbs2))

def is_in(bbs1,bbs2):
    return contains(bbs2,bbs1)

def is_not_in(bbs1,bbs2):
    return np.logical_not(is_in(bbs1,bbs2))


# generations of data for negative examples and generic rectangles used to feed the variables x,y,z

nr_random_bbs = 50
def get_data(type):
    feed_dict = {}
    feed_dict["?left_xy"] = left_data
    feed_dict["?right_xy"] = right_data
    feed_dict["?below_xy"] = below_data
    feed_dict["?above_xy"] = above_data
    feed_dict["?contains_xy"] = non_contain_data[np.random.choice(len(non_contain_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?contained_in_xy"] = non_in_data[np.random.choice(len(non_in_data),nr_random_bbs,replace=True)].astype(np.float32)
        

    feed_dict["?non_left_data"] = non_left_data[np.random.choice(len(non_left_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?not_left_xy"] = non_right_data[np.random.choice(len(non_right_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?not_below_xy"] = non_below_data[np.random.choice(len(non_below_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?not_above_xy"] = non_above_data[np.random.choice(len(non_above_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?not_contains_xy"] = non_contain_data[np.random.choice(len(non_contain_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?not_contained_in_xy"] = non_in_data[np.random.choice(len(non_in_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?x"] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?y"] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict["?z"] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    return feed_dict
