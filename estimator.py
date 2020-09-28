import math
from quaternion import eul2quat

#  Angles measure from center image
#   + pitch angle when target is closer to top
#   + yaw angle when target is right from center
#   target yaw and pitch values
# camfovh -- degree
# const float CameraFOVH_rad = MICROSOFT_FOVH * M_PI / 180;
# const float targetsize = 0.2; // meter


def estimate_yaw_pitch_dist(camfovh, targ_sz, im_w, im_h, left, top, right, bot, euler):

    x, y, z = euler
    camfovh_rad = camfovh * math.pi / 180
    camerafovv_rad = camfovh_rad * im_h / im_w # werical angle in radian

    #multiplier based on constants and camera settings
    m = im_w * targ_sz * 2 * math.tan(camerafovv_rad / 2)    
    w = right - left

    distance =  m / w

    x_offs2x = left + right - im_w
    y_offs2x = top + bot - im_h

    
    z = math.atan2(x_offs2x * math.tan(camfovh_rad / 2), im_w) # yaw
    y = math.atan2(y_offs2x * math.tan(camfovh_rad / 2), im_h) # pitch
    x = 0; #// camera roll

    return eul2quat((x, y, z)), distance

