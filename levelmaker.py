import json
import numpy as np


def gridlevel(xlen, ylen , radius, dither=0, radius_white=50, fname="grid" ):

    screen = [1000, 800]
    centers_red = []
    for x in range( screen[0]//xlen, screen[0]+1,  screen[0]//xlen ):
        for y in range( screen[1]//ylen, screen[1]+1,  screen[1]//ylen  ):
            if dither:
                dither_x = np.random.randint( -dither, dither )
                dither_y = np.random.randint( -dither, dither )
                centers_red.append((x+dither_x, y+dither_y))
            else:
                dither_x, dither_y = 0,0


    level = Level(
        center_white=[1000, 600],
        radii_red=(radius, )*xlen*ylen,
        radius_white=radius_white,
        centers_red=centers_red
    )
    with open("levels/{}".format(fname), 'w' ) as fd:
        json.dump(level, fd, indent=2)
    return level


def Level( start_pos=None, start_angle=None, **kwargs ):
    level = kwargs
    if "ship" not in level:
        level["ship"] = {}
    if start_pos is None:
        level["ship"]["starting_pos"] = [20, 50]
    if start_angle is None:
        level["ship"]['starting_angle'] = 45


    return level


def circlelevel( cradius, pradius, nplanets ):

    thetas = np.arange(0, 2*np.pi, 2*np.pi/nplanets)
    centers_red = []
    for theta in thetas:
        x = 500+cradius*np.cos( theta )
        y = 400+cradius*np.sin(theta )
        centers_red.append((x, y ))

    level = Level(
        centers_red=centers_red,
        center_white=[500,400],
        radii_red=(pradius,)*len(thetas),
        radius_white = 50)
    with open("levels/circle", 'w') as fd:
        json.dump(level, fd)
    return level


gridlevel(7,5, 40, 100, 40, "grid2" )
#circlelevel(300, 20, 40 )
