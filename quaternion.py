import math
# input 4 dimensional quaternion (w,x,y,z)
# returns normalized quaternion
def quat_norm(quaternion):

    w, x, y, z = quaternion
    # calc vector size
    d = (w**2 + x**2 + y**2 + z**2)**0.5

    w /= d
    x /= d
    y /= d
    z /= d

    return w, x, y, z

# degrees in radian
def quat2eul(quaternion):

    quaternion = quat_norm(quaternion)
    w, x, y, z = quaternion
    
    ez = math.atan2(2 * (x*y + w*z, w*w + x*x - y*y - z*z))
    ey = math.asin(-2 * (x*z - w*y))
    ex = math.atan2(2 * (y*z + w*x, w*w - x*x - y*y + z*z))

    return ex,ey,ez

# degrees in radian
def eul2quat(euler):

    x,y,z = euler
    cx = math.cos(x / 2)
    cy = math.cos(y / 2)
    cz = math.cos(z / 2)

    sx = math.sin(x / 2)
    sy = math.sin(y / 2)
    sz = math.sin(z / 2)

    w = cz * cy * cx + sz * sy * sx
    x = cz * cy * sx - sz * sy * cx
    y = cz * sy * cx + sz * cy * sx
    z = sz * cy * cx - cz * sy * sx

    return w, x, y, z

def quat_prod(quaternion1, quaternion2):

    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2

    w = w1 * w2
    x = x1 * x2
    y = y1 * y2
    z = z1 * z2
    return w, x, y, z


def quat_mul(quaternion1, quaternion2):

    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2

    w = -x1*x2 - y1*y2 - z1*z2 + w1*w2
    x =  x1*w2 + y1*z2 - z1*y2 + w1*x2
    y = -x1*z2 + y1*w2 + z1*x2 + w1*y2
    z =  x1*y2 - y1*x2 + z1*w2 + w1*z2
    return w, x, y, z

def quat_add(quaternion1, quaternion2):

    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2

    w = w1 + w2
    x = x1 + x2
    y = y1 + y2
    z = z1 + z2

    return w, x, y, z


def quat_conj(quaternion):
    
    w, x, y, z = quaternion
    return w, -x, -y, -z

def quat_scale(quaternion, d):
    
    w, x, y, z = quaternion
    return w*d, x*d, y*d, z*d


def eul_conj(euler):

    x, y, z = euler

    if x < 0: x += math.pi
    else: x -= math.pi

    if y < 0: y += math.pi
    else: y -= math.pi

    if z < 0: z += math.pi
    else: z -= math.pi

    return x, -y, z

def quat_msqe(quaternion1, quaternion2):
    
    w1, x1, y1, z1 = quaternion1
    w2, x2, y2, z2 = quaternion2

    return ((w1-w2)**2 + (x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) / 4


def eul_msqe(euler1, euler2):

    x1, y1, z1 = euler1
    x2, y2, z2 = euler2
    
    return ((x1-x2)**2 + (y1-y2)**2 + (z1-z2)**2) / 3
    
