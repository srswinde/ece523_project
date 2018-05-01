#!/usr/bin/env python

from copy import deepcopy
from sklearn.neural_network import MLPClassifier
import pygame
import numpy as np
import math
#import pickle
import time
import pickle, datetime
VEC = pygame.math.Vector2


class colors:
    white = (255,)*3
    red = (255, 0, 0)
    green = (0, 255, 0)
    blue = (0, 255, 0)
    black = (0, 0, 0)


config = dict(
    planet_radius=75,
    gravity= 0,  #-0.002,
    land_angle=10,
    land_speed=0.25,
    delta_angle=2,
    thrust=0.01,
    dt=5, #0.05
    flat_index = 0,
    num_ships = 30,
    planet_center = VEC( 700, 500 ),
    planet_center2 = VEC( 100, 100 ),
    speed_multiplier = 50.35,
    time_limit = 6
)

#Neural Network Structure
n_inputs = 3
n_hidden = 4
n_output = 1

#These are used for initializing the scikitlearn Neural Networks
X = np.zeros(n_inputs)
X_train = np.array([X,X])
y_train = np.array(range(n_output+1))  #np.array([0,1])



class PygView( object ):
    def __init__(self, width=1000, height=1000, fps=60):
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

        #config["planet_center"] = VEC( self.width//2, self.height//2 )
        self.landing_points = self.do_planet(
            radius=config["planet_radius"],
            center=config['planet_center'],
            flat_index = config['flat_index'])
        self.ship = space_ship( self.screen, self.landing_points )
        self.ships = []
        for i in range(config['num_ships']):
            self.ships.append(space_ship( self.screen, self.landing_points ))
        self.Newships = []
        self.game_over = False
        self.stop_printing = False
        self.generation = 0
        self.bestScore = 0
        self.prevShips = []
        self.prevFitness = []
        self.logLst = []

    def reset(self):
        self.ship = space_ship( self.screen, self.landing_points )
        #self.sp = space_ship( self.screen, self.landing_points )
        self.game_over = False

    def run(self):
        """The mainloop
        """
        running = True
        ai_key = "none"
        count = 0
        self.ship = self.ships[0] 
        while running:
            da = 0
            thrust = 0.0
            #initialize ship     
            '''self.ship.physics(
                delta_angle=da,
                thrust=thrust,
                stop=self.ship.crashed)'''
            self.do_planet(
                radius=config["planet_radius"],
                center=config['planet_center'],
                flat_index = config['flat_index'])
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    all_crashed = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                        all_crashed = True
                    if event.key == pygame.K_r:
                        self.reset()
                        #self.resetShipLocs()
            
            keys = pygame.key.get_pressed()

            da = 0
            thrust = 0.0
            if keys[pygame.K_LEFT]:
                da = -config["delta_angle"]
            if keys[pygame.K_RIGHT]:
                da = config["delta_angle"]
            if keys[pygame.K_UP]:
                thrust = config["thrust"]
            # Do the physics on the spaceship
            self.ship.physics(
                delta_angle=da,
                thrust=thrust,
                stop=self.ship.crashed )
            # Did we land?
            if(self.ship.check_on_planet() or self.ship.check_pos_screen()==False):
                self.ship.crashed = True

            dist = self.circleIntercept()
            if(dist == -1):
                dist =self.wallIntercept()

            self.draw_text("Tip:{} Back:{} dist{}".format(self.ship.tip,self.ship.back,dist))

            pygame.display.flip()
            self.screen.blit( self.background, (0, 0) )

            #Run this to update fitness
            #self.ships[j].updateFitness()


            #self.updateWeights()
            #self.resetShipLocs()
        pygame.quit()

    def wallIntercept(self):
        #m is the slope of the line. Used to describe line in direction of ship
        direction = 4
        
        if(direction ==0):
            #straight
            m = self.ship.tip - self.ship.back  
            x, y = self.ship.tip[0],self.ship.tip[1]          
        if(direction == 1):
            #left
            m = self.ship.left - self.ship.right
            x, y = self.ship.left[0],self.ship.left[1]        
        if(direction == 2):
            #right
            m = self.ship.right - self.ship.left
            x, y = self.ship.right[0],self.ship.right[1]       
        if(direction == 3):
            #left-staight
            m = (self.ship.left + self.ship.tip)/2 - self.ship.right 
            x, y = ((self.ship.left + self.ship.tip)/2)[0],((self.ship.left + self.ship.tip)/2)[1]
        if(direction == 4):
            #right-straight
            m = (self.ship.right + self.ship.tip)/2 - self.ship.left 
            x, y = ((self.ship.right + self.ship.tip)/2)[0],((self.ship.right + self.ship.tip)/2)[1]
        #Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if(m[0]==0):
            m=999999
        else:
            m = m[1]/m[0]   

        """ m_lw = the slope of the line that describes the left wall of the game world  """
        m_lw = 999999
        m_rw = 999999
        m_bw = 0
        m_tw = 0

        """rw = rightWall, lw = leftWall, bw = bottomWall, tw = topWall"""
        x_lw, y_lw = 0,0
        x_rw, y_rw = 1000,0
        x_bw, y_bw = 1000,800
        x_tw, y_tw = 1000,0

        m_walls = [m_lw,m_rw,m_bw,m_tw]
        x_w = [x_lw,x_rw,x_bw,x_tw]
        y_w = [y_lw,y_rw,y_bw,y_tw]

        lDistances = []
        for ii in range(4):
            if(m - m_walls[ii] == 0):
                x_i = 999999
            else:
                x_i = (m*x - y - m_walls[ii]*x_w[ii] + y_w[ii])/(m - m_walls[ii])

            y_i = m*(x_i - x) + y

            dist = (VEC(x_i,y_i) - VEC(x, y)).length()

            if(dist != -1):
                if(direction == 0):
                    #straight
                    if (self.ship.back - VEC(x_i,y_i)).length() < (self.ship.tip - VEC(x_i,y_i)).length():
                        dist = -1   
                if(direction == 1):
                    #left
                    if (self.ship.right - VEC(x_i,y_i)).length() < (self.ship.left - VEC(x_i,y_i)).length():
                        dist = -1    
                if(direction == 2):
                    #right
                    if (self.ship.left - VEC(x_i,y_i)).length() < (self.ship.right - VEC(x_i,y_i)).length():
                        dist = -1      
                if(direction == 3):
                    #left-staight
                    if (self.ship.right - VEC(x_i,y_i)).length() < ((self.ship.left + self.ship.tip)/2 - VEC(x_i,y_i)).length():
                        dist = -1  
                if(direction == 4):
                    #right-straight
                    if (self.ship.left - VEC(x_i,y_i)).length() < ((self.ship.right + self.ship.tip)/2 - VEC(x_i,y_i)).length():
                        dist = -1                   
            if(dist != -1):
                lDistances.append(dist)
            
        return np.min(lDistances)

    def circleIntercept(self):
        """https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle"""
        
        #m is the slope of the line. c is the y intercept. used to describe line in direction of ship
        direction = 4
        
        if(direction ==0):
            #straight
            m = self.ship.tip - self.ship.back  
            lineStart = self.ship.tip          
        if(direction == 1):
            #left
            m = self.ship.left - self.ship.right
            lineStart = self.ship.left       
        if(direction == 2):
            #right
            m = self.ship.right - self.ship.left     
            lineStart = self.ship.right 
        if(direction == 3):
            #left-staight
            m = (self.ship.left + self.ship.tip)/2 - self.ship.right 
            lineStart = (self.ship.left + self.ship.tip)/2
        if(direction == 4):
            #right-straight
            m = (self.ship.right + self.ship.tip)/2 - self.ship.left 
            lineStart = (self.ship.right + self.ship.tip)/2
        #Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if(m[0]==0):
            m=999999
        else:
            m = m[1]/m[0]     

        #We want left and right 'seeing directions' to be at the back of the ship
        c = lineStart[1] - m * lineStart[0]


        p = config['planet_center'][0]
        q = config['planet_center'][1]
        r = config['planet_radius']

        A = m**2 + 1
        B = 2*(m*c - m*q - p)
        C = q**2-r**2+p**2-2*c*q+c**2
        
        #If B^2−4AC<0 then the line misses the circle
        #If B^2−4AC=0 then the line is tangent to the circle.
        #If B^2−4AC>0 then the line meets the circle in two distinct points.
        if(B**2 - 4*A*C < 0):
            x = -1
            y = -1
            dist = -1
        elif(B**2 - 4*A*C == 0 ):
            x = -B/(2*A)
            y = m*x + c
            dist = (VEC(x,y) - VEC(lineStart[0],lineStart[1])).length()
        else:
            x1 = (-B + np.sqrt(B**2 - 4*A*C))/(2*A)
            x2 = (-B - np.sqrt(B**2 - 4*A*C))/(2*A)
            y1 = m*x1 + c
            y2 = m*x2 + c

            l1 = (VEC(x1,y1) - VEC(lineStart[0],lineStart[1])).length()
            l2 = (VEC(x2,y2) - VEC(lineStart[0],lineStart[1])).length()

            #Pick the point on the circle that is closest to the ship
            if(l1 < l2): 
                x = x1
                y = y1
                dist = l1
            else:
                x = x2
                y = y2
                dist = l2
        
        #Check to make sure the line intercepts the circle on the front side of the ship
        if(dist != -1):
            if(direction ==0):
                #straight
                if (self.ship.back - VEC(x,y)).length() < (self.ship.tip - VEC(x,y)).length():
                    dist = -1          
            if(direction == 1):
                #left
                if (self.ship.right - VEC(x,y)).length() < (self.ship.left - VEC(x,y)).length():
                    dist = -1      
            if(direction == 2):
                #right
                if (self.ship.left - VEC(x,y)).length() < (self.ship.right - VEC(x,y)).length():
                    dist = -1      
            if(direction == 3):
                #left-staight
                if (self.ship.right - VEC(x,y)).length() < ((self.ship.left + self.ship.tip)/2 - VEC(x,y)).length():
                    dist = -1  
            if(direction == 4):
                #right-straight
                if (self.ship.left - VEC(x,y)).length() < ((self.ship.right + self.ship.tip)/2 - VEC(x,y)).length():
                    dist = -1             
        return dist


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

    def resetShipLocs(self):
        """Reset the ship locations, but not their neural net weights"""

        for i in range(config['num_ships']):
            self.ships[i].pos = VEC((150, 150))
            self.ships[i].angle = 90
            self.ships[i].velocity = VEC(0, 0)
            self.ships[i].crashed = False
            #self.ship.fitness = 0

class space_ship:
    """The space shipe class"""
    def __init__(self, screen, landing_points, pos=(150, 150), angle=90 ):
        self.pos = VEC( pos )
        self.angle = angle
        self.screen = screen
        self.velocity = VEC(0, 0)
        self.landing_points = landing_points
        self.crashed = False
        self.fitness = 0
        self.inputs = np.zeros(n_inputs)

        # find mid point of landing
        li = landing_points.shape[0]//2
        self.mid_landing_point = VEC(list(self.landing_points[li]))

        # VEC can't be instantiated with array
        # so we convert to list
        lp0 = VEC(list(self.landing_points[0])) -  config["planet_center"]
        lpf = VEC(list(self.landing_points[-1])) - config["planet_center"]
        self.la0 = lp0.angle_to(VEC(1, 0))
        self.laf = lpf.angle_to(VEC(1, 0))



        self.mlp = MLPClassifier(hidden_layer_sizes=(n_hidden),max_iter=1, activation = "tanh")
        self.mlp.fit(X_train,y_train)
        #Initialize the MLP with random weights

        self.mlp.intercepts_[0] = np.random.rand(n_hidden)*2-1
        self.mlp.intercepts_[1] = np.random.rand(n_output)*2-1
        self.mlp.coefs_[0] = np.random.rand(n_inputs,n_hidden)*2-1
        self.mlp.coefs_[1] = np.random.rand(n_hidden,n_output)*2-1
        self.minDLandStrip = None
        self.debug = False

   
    def render(self, color ):

        tip = VEC( 10, 0)
        left = VEC(-5, -5)
        right = VEC(-5, 5)

        for pt in (tip, right, left):
            pt.rotate_ip( self.angle )
            pt += self.pos
        pygame.draw.polygon(self.screen, color, ( tip, left, right ) )
        self.back = (left + right)/2

        self.tip, self.left, self.right = tip, left, right

    def physics( self, thrust=0.0, delta_angle=0.0, stop=False ):
        ppos =  config["planet_center"]

        gravity = config["gravity"]*(self.pos-ppos).normalize()
        dt = config["dt"]
        if not stop:
            thrust_vector = VEC(1, 0).rotate(self.angle)*thrust
            # self.velocity = self.velocity + (gravity+thrust_vector)*dt
            self.velocity = config['speed_multiplier']*thrust_vector

            self.pos = self.pos + self.velocity*dt
            self.angle += delta_angle

        if thrust == 0:
            color = colors.green
        else:
            color = colors.red

        if self.minDLandStrip is None:
            self.minDLandStrip = (self.pos - self.mid_landing_point).length()
        elif self.minDLandStrip > (self.pos - self.mid_landing_point).length():
            self.minDlandStrip = (self.pos - self.mid_landing_point).length()

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
        #print(self.la0," ", self.laf, " " ,  planet_angle )
        if self.la0 <= planet_angle <= self.laf \
                or self.laf <= planet_angle <= self.la0:
            return True

        else:
            return False

    def check_pos_screen(self):
        if(self.pos[0] > 0 and self.pos[0] < 1000 and self.pos[1] > 0 and self.pos[1] < 800):
            return True
        else:
            return False
        #print(self.pos)

    def check_on_planet(self):
        # if any part of the ship is touching the planet
        # we have landed
        for pt in (self.tip, self.left, self.right):

            if (pt - config["planet_center"]).length()\
                    < config["planet_radius"]:
                return True
        return False

    def NN_Inputs(self):
        ship_coors = self.tip
        land_coors = self.landing_points[0]

        ship_angle = self.angle%360
        dSurface = (ship_coors - config["planet_center"]).length() - config["planet_radius"]
        dLandStrip = (ship_coors - land_coors).length()

        #Normalize inputs, want -1 to 1 range
        ship_angle = ship_angle/360 * 2 - 1
        maxD = VEC(1000,800).length()
        dSurface = dSurface/maxD*2 - 1
        dLandStrip = dLandStrip/maxD*2 - 1


        return ship_angle,dSurface,dLandStrip



# End win condition methods.
if __name__ == '__main__':

    #If we want to run the game without rendering to the screen
    PygView(1000, 800).run()
        