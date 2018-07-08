import tensorflow as tf
import logictensornetworks as ltn
import numpy as np
from logictensornetworks import Forall,Exists, Equiv, Implies, And, Or, Not
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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

# and of data generations

# start the definition of the language:

# variables for pairs of rectangles ....

# ... for positive examples of every relation
lxy = ltn.variable("left_xy", tf.cast(left_data, tf.float32))
rxy = ltn.variable("right_xy",tf.cast(right_data,tf.float32))
bxy = ltn.variable("below_xy",tf.cast(below_data,tf.float32))
axy = ltn.variable("above_xy",tf.cast(above_data,tf.float32))
cxy = ltn.variable("contains_xy",tf.cast(contain_data,tf.float32))
ixy = ltn.variable("in_xy", tf.cast(in_data, tf.float32))

# ... for negative examples (they are placeholders which are filled with data
# randomly generated every 100 trian epochs

nlxy = ltn.variable("not_left_xy", 8)
nrxy = ltn.variable("not_right_xy",8)
nbxy = ltn.variable("not_below_xy",8)
naxy = ltn.variable("not_above_xy",8)
ncxy = ltn.variable("not_conts_xy",8)
nixy = ltn.variable("not_is_in_xy",8)

# printing out the cardinality of examples

pxy = [lxy, rxy, bxy, axy, cxy, ixy]
npxy = [nlxy,nrxy,nbxy,naxy,ncxy,nixy]

for xy in pxy:
    print(xy.name,xy.shape)

# variables for single rectangles

x = ltn.variable("x",4)
y = ltn.variable("y",4)
z = ltn.variable("z",4)

# a rectangle and a set of rectangle used to show the results

ct = ltn.constant("ct",[.5,.5,.3,.3])
t = ltn.variable("t",tf.cast(bbst,tf.float32))


# relational predicates

L = ltn.predicate("left",8)
R = ltn.predicate("right",8)
B = ltn.predicate("below",8)
A = ltn.predicate("above",8)
C = ltn.predicate("contains",8)
I = ltn.predicate("in",8)

P = [L,R,B,A,C,I]

inv_P = [R,L,A,B,I,C]

# constraints/axioms

constraints =  [Forall(pxy[i],P[i](pxy[i]))
                for i in range(6)]
constraints += [Forall(npxy[i],Not(P[i](npxy[i])))
                for i in range(6)]
constraints += [Forall((x,y),Implies(P[i](x,y),inv_P[i](y,x)))
                for i in range(6)]
constraints += [Forall((x,y),Not(And(P[i](x,y),P[i](y,x))))
                for i in range(6)]
# constraints += [Forall((x,y,z),Implies(I(x,y),Implies(P[i](y,z),P[i](x,z)))) for i in range(6)]


loss = -tf.reduce_min(tf.concat(constraints,axis=0))
opt = tf.train.AdamOptimizer(0.05).minimize(loss)
init = tf.global_variables_initializer()


# generations of data for negative examples and generic rectangles used to feed the variables x,y,z

nr_random_bbs = 50
def get_feed_dict():
    feed_dict = {}
    feed_dict[nlxy] = non_left_data[np.random.choice(len(non_left_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[nrxy] = non_right_data[np.random.choice(len(non_right_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[nbxy] = non_below_data[np.random.choice(len(non_below_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[naxy] = non_above_data[np.random.choice(len(non_above_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[ncxy] = non_contain_data[np.random.choice(len(non_contain_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[nixy] = non_in_data[np.random.choice(len(non_in_data),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[x] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[y] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    feed_dict[z] = bbs12[np.random.choice(len(bbs12),nr_random_bbs,replace=True)].astype(np.float32)
    return feed_dict




with tf.Session() as sess:

# training:

    sess.run(init)
    feed_dict = get_feed_dict()
    for i in range(10000):
        sess.run(opt,feed_dict=feed_dict)
        if i % 100 == 0:
            sat_level=sess.run(-loss, feed_dict=feed_dict)
            print(i, "sat level ----> ", sat_level)
            if sat_level > .99:
                break

# evaluate the truth value of a formula ....
    print(sess.run([Forall((x,y,z),Implies(I(x,y),
                                           Implies(P[i](y,z),P[i](x,z))))
                    for i in range(6)],feed_dict=feed_dict))

# evaluate the truth value of P(ct,t) where ct is a central rectangle, and
# t is a set of rectangles randomly generated.

    preds = sess.run([X(ct,t) for X in P])

# plotting the value of the relation, on the centroid of t.

    fig = plt.figure(figsize=(12,8))
    jet = cm = plt.get_cmap('jet')
    cbbst = bbst[:,:2] + 0.5*bbst[:,2:]
    for j in range(6):
        plt.subplot(2, 3, j + 1)
        plt.scatter(cbbst[:,0], cbbst[:,1], c=preds[j][:, 0])
    plt.show()

