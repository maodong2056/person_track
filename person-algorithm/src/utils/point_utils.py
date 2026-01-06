import numpy as np
import math
# from scipy.linalg import solve

def tanAngle(point1, point2, point3):
    """
    :param point1: shoulder/neck piont in np.array of shape(3, )!! This point must be the angle point!
    :param point2: plevis piont in np.array of shape(3, )
    :param point3: hands piont in np.array of shape(3, )
    :return: Cosine of the angle between the hand and the body
    """
    assert point1.shape[0] == 3
    assert point2.shape[0] == 3
    assert point3.shape[0] == 3

    a = point2 - point1
    b = point3 - point1

    cos_ab = a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b))
    sin_ab = np.sqrt((1 - cos_ab**2))
    tan_ab = sin_ab/cos_ab
    # print(sin_ab, cos_ab, tan_ab)
    return sin_ab, cos_ab, tan_ab


def disAndAngle(point1, point2, point3, radius, dist, bias):
    """
    Top view see hand (body is center point of circle), so we set all of yi=0;
    :param point1: body, [x1, 0, z1]
    :param point2: Sweeper/camera, [x1, 0, 0]
    :param point3: hand point that project to floor [x2, 0, z2]
    :param radius:
    :param dist: Sweeper/camera to body
    :return: where to go
    """
    point2[0] = point2[0] - bias
    _, cos_center, _ = tanAngle(point1, point2, point3)

    dist_new = np.sqrt(bias**2 + dist**2)

    moving_dist = np.sqrt(radius**2 + dist_new**2 - 2 * radius * dist_new * cos_center)
    # moving_direction = math.degrees(math.acos((moving_dist**2 + dist**2 - radius**2) / (2 * moving_dist * moving_dist)))
    moving_direction = math.acos((moving_dist**2 + dist_new**2 - radius**2) / (2 * moving_dist * dist_new))
    moving_direction = (moving_direction * 180.) / np.pi
    return moving_dist, moving_direction

if __name__ == '__main__':
    # res = disAndAngle((0,0,0), (0,0,np.sqrt(3)), (1,0,np.sqrt(3)))
    _, cos_center, _ = tanAngle(np.array([170,0,3980]), np.array([0,0,0]),
                                np.array([-133,0,3512]))
    print((np.arccos(cos_center)/np.pi)*180)