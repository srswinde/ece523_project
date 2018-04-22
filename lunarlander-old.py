#!/usr/bin/env python

import pygame
import numpy as np
import math
VEC = pygame.math.Vector2


class colors:
    white = (255,)*3
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 255, 0)
    black = (0, 0, 0)


config = dict(
    planet_radius=75,
    gravity=-0.002,
    land_angle=10,
    land_speed=0.25,
    delta_angle=1,
    thrust=0.01,
    dt=0.50
)


class PygView( object ):

    def __init__(self, width=1000, height=1000, fps=30):
        """Initialize pygame, window, background, font,...
        """
        pygame.init()
        pygame.display.set_caption("Press ESC to quit")
        self.width = width
        self.height = height
        # self.height = width // 4
        self.screen = pygame.display.set_mode(
            (self.width, self.height),
            pygame.DOUBLEBUF)
        self.background = pygame.Surface(self.screen.get_size()).convert()
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.font = pygame.font.SysFont('mono', 20, bold=True)

        config["planet_center"] = VEC( self.width//2, self.height//2 )
        self.landing_points = self.do_planet(
            radius=config["planet_radius"],
            center=config['planet_center'] )
        self.sp = space_ship( self.screen, self.landing_points )
        self.game_over = False

    def reset(self):

        self.sp = space_ship( self.screen, self.landing_points )
        self.game_over = False

    def run(self):
        """The mainloop
        """
        running = True
        while running:
            da = 0
            thrust = 0.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        self.reset()

            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT]:
                da = -config["delta_angle"]

            if keys[pygame.K_RIGHT]:
                da = config["delta_angle"]

            if keys[pygame.K_UP]:
                thrust = config["thrust"]

            # Render the planet
            self.do_planet(
                radius=config["planet_radius"],
                center=config['planet_center'])

            # Do the physics on the spaceship
            self.sp.physics(
                delta_angle=da,
                thrust=thrust,
                stop=self.game_over )

            # Did we land?
            if self.sp.check_on_planet():
                self.game_over = True

                if self.sp.check_orientation() \
                        and self.sp.check_land_spot() \
                        and self.sp.check_speed():
                    self.draw_text("YOU LANDED SUCCESSFULLY!")
                else:
                    self.draw_text("YOU CRASHED!")

            # Not yet update the message on the screen
            else:

                self.draw_text(
                    "Orient:{}  Land:{} speed:{}".format(
                        self.sp.check_orientation(),
                        self.sp.check_land_spot(),
                        self.sp.check_speed()
                    ))

            pygame.display.flip()
            self.screen.blit( self.background, (0, 0) )

        pygame.quit()

    def draw_text( self, text ):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render( text, True, (0, 255, 0) )
        # // makes integer division in python3
        self.screen.blit(
            surface, ( ( self.width - fw ), ( self.height - fh )) )

    def do_planet(self, radius, center=(200, 200), flat_index=0):
        """Draw the planet including the gaussian noise
        to simulate erruptions"""

        # angle in radians between points defining the planet
        res = 0.01

        # numer of points defining the planet
        npoints = int( 2*math.pi//res + 1)
        thetas = np.arange(0, 2*math.pi, res)
        plist = np.zeros((npoints, 2))

        # the landing part of the planet
        fi0 = flat_index % npoints
        fi1 = ( flat_index + npoints//10 ) % npoints

        landform = np.random.normal( scale=2, size=( npoints, 2) )
        landform[ fi0:fi1, : ] = 0
        plist[:, 0] = center[0] + radius*np.cos( thetas )
        plist[:, 1] = center[1] + radius*np.sin( thetas )

        pygame.draw.polygon( self.screen, colors.white, plist + landform )
        return plist[ fi0:fi1, : ]


class space_ship:
    """The space shipe class"""
    def __init__(self, screen, landing_points, pos=(150, 150), angle=90 ):
        self.pos = VEC( pos )
        self.angle = angle
        self.screen = screen
        self.velocity = VEC(0, 0)
        self.landing_points = landing_points

        # VEC can't be instantiated with array
        # so we convert to list
        lp0 = VEC(list(self.landing_points[0])) - config["planet_center"]
        lpf = VEC(list(self.landing_points[-1])) - config["planet_center"]
        self.la0 = lp0.angle_to(VEC(1, 0))
        self.laf = lpf.angle_to(VEC(1, 0))

    def render(self, color ):

        tip = VEC( 10, 0)
        left = VEC(-5, 5)
        right = VEC(-5, -5)

        for pt in (tip, right, left):
            pt.rotate_ip( self.angle )
            pt += self.pos

        pygame.draw.polygon(
            self.screen, color, ( tip, left, right ) )
        self.tip, self.left, self.right = tip, left, right

    def physics( self, thrust=0.0, delta_angle=0.0, stop=False ):
        ppos = config["planet_center"]

        gravity = config["gravity"]*(self.pos-ppos).normalize()
        dt = config["dt"]
        if not stop:
            thrust_vector = VEC(1, 0).rotate(self.angle)*thrust
            self.velocity = self.velocity + (gravity+thrust_vector)*dt
            self.pos = self.pos + self.velocity*dt
            self.angle += delta_angle
        if thrust == 0:
            color = colors.green
        else:
            color = colors.red

        self.render( color )

# Begin methods to check win conditions
    def check_orientation(self):
        pangle = ((self.left - self.right).angle_to(
            self.pos-config["planet_center"]))

        if pangle > -90-config["land_angle"] \
                and pangle < -90+config["land_angle"]:

            return True
        else:
            return False

    def check_speed(self):
        if self.velocity.length() < config["land_speed"]:
            return True
        else:
            return False

    def check_land_spot( self ):
        planet_angle = (self.pos - config["planet_center"]).angle_to(VEC(1, 0))

        if self.la0 <= planet_angle <= self.laf \
                or self.laf <= planet_angle <= self.la0:
            return True

        else:
            return False

    def check_on_planet(self):

        # if any part of the ship is touching the planet
        # we have landed
        for pt in (self.tip, self.left, self.right):

            if (pt - config["planet_center"]).length()\
                    < config["planet_radius"]:
                return True

        return False


# End win condition methods.
if __name__ == '__main__':

    # call with width of window and fps
    PygView(1000, 800).run()
