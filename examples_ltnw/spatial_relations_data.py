import numpy as np

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

def generate_data(nr_rectangle_pairs,fun):
    """ Generates number of rectangles for which fun is true """
    results=None
    while results is None or len(results) < nr_rectangle_pairs:
        candidate_rectangles_1=generate_rectangles(100 * nr_rectangle_pairs)
        candidate_rectangles_2=generate_rectangles(100 * nr_rectangle_pairs)
        ok=fun(candidate_rectangles_1,candidate_rectangles_2)
        ok_rectangles=np.concatenate([candidate_rectangles_1[ok],candidate_rectangles_2[ok]],axis=1)
        if results == None:
            results = ok_rectangles 
        results=np.concatenate([results,ok_rectangles],axis=0)
    return results[:nr_rectangle_pairs,:]
    