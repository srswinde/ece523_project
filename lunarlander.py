#!/usr/bin/env python

from sklearn.neural_network import MLPClassifier
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
    dt=0.05,
    flat_index = 300,

    # where on the planet (in degrees)
    # does the flat part start
    flat_angle_start = 90,
    # Width in degrees of the flat part
    flat_angle_width = 30,
    reference_vector = VEC(1, 0),

    num_ships = 1
)

#These are just used for initializing the scikitlearn Neural Networks
X_train = np.array([[0,0,0],[0,0,0],[0,0,0],[0,0,0]])
y_train = np.array([0,3,1,2])



def angle_to(vec1, vec2=None):
    if vec2 is None:
        vec2 = config["reference_vector"]


    return (vec1.angle_to(vec2) + 360) % 360

def fix_angle(angle):
    return (angle+360) % 360


def angle_between( test_angle, angle, delta_angle ):

    """Check if the test_angle is between angle and
    angle + delta_angle this. This assumes angles
    are in degrees"""

    assert( delta_angle > 0 )
    test_angle = fix_angle( test_angle )
    angle1 = fix_angle( angle )
    angle2 = fix_angle( angle + delta_angle )

    # The normal case
    if angle1 < angle2:
        return (angle1 <= test_angle <= angle2)

    # Angle1 is the greater than angle2
    # because angle2 was modulused by 360
    else:
        return (angle1 <= test_angle <= angle2+360)


class PygView( object ):

    def NeuralNet(self,ship_angle,dSurface,dLandStrip):



        mlp = MLPClassifier(hidden_layer_sizes=(4),max_iter=1)
        mlp.fit(X_train,y_train)
        mlp.coefs_[0] = np.random.rand(3,4)*2-1
        mlp.coefs_[1] = np.random.rand(4,4)*2-1

        string_output = "none"

        X = np.array([ship_angle,dSurface,dLandStrip])
        output = mlp.predict(X.reshape(1,-1))

        if(output==0):
            string_output = "none"
        elif(output==1):
            string_output = "left"
        elif(output==2):
            string_output = "right"
        elif(output==3):
            string_output = "up"

        return string_output



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
            center=config['planet_center'])

        self.ships = []
        for i in range(config['num_ships']):
            self.ships.append(space_ship( self.screen, self.landing_points ))

        self.game_over = False
        self.stop_printing = False

    def reset(self):
        for i in range(config['num_ships']):
            self.ships[i] = space_ship( self.screen, self.landing_points )
        #self.sp = space_ship( self.screen, self.landing_points )
        self.game_over = False

    def run(self):
        """The mainloop
        """
        running = True
        ai_key = "none"
        while running:

            da = 0  # angle of rotation
            thrust = 0.0
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    if event.key == pygame.K_r:
                        self.reset()
                        #self.resetShipLocs()

            keys = pygame.key.get_pressed()

            # Render the planet
            self.do_planet(
                radius=config["planet_radius"],
                center=config['planet_center'])

            All_Crashed = True
            for i in range(config['num_ships']):
                '''
                x =  np.random.randint(0,3)
                x = 4
                if(x == 0):
                   ai_key = "left"
                elif(x==1):
                    ai_key = "right"
                elif(x==2):
                    ai_key = "up"
                '''
                ai_key = "none"
                if keys[pygame.K_LEFT] or ai_key == "left":
                    da = -config["delta_angle"]

                if keys[pygame.K_RIGHT] or ai_key == "right":
                    da = config["delta_angle"]

                if keys[pygame.K_UP] or ai_key == "up":
                    thrust = config["thrust"]

                # Do the physics on the spaceship
                self.ships[i].physics(
                    delta_angle=da,
                    thrust=thrust,
                    stop=self.ships[i].crashed )

                '''
                if(self.ships[i].check_pos_screen()==False):
                    #self.game_over = True
                    self.draw_text("YOU DIE! FOOL!")
                    self.ships[i].crashed = True
                '''

                #ship_angle,dSurface,dLandStrip = self.ships[i].NN_Inputs()
                #print("Fitness: ", dLandStrip)
                ai_key = self.ships[i].predict()
                #ai_key = self.NeuralNet(1,2,3)

                # Did we land?
                if( self.ships[i].check_on_planet() ):
                    if self.ships[i].check_land_spot:
                        print("Landed!")
                    else:
                        print("crashed on planet")

                if( self.ships[i].check_pos_screen()==False ):
                    self.ships[i].crashed = True

            #Check if all the ships have crashed
                if(self.ships[i].crashed == False):
                    All_Crashed = False
            #If all the ships have crashed, evaluate all their fitness functions. Reset.
            if(All_Crashed == True):
                self.game_over = True
                self.updateWeights()
                #self.reset()


                '''
                if self.ships[i].check_land_spot():
                    self.draw_text("YOU LANDED SUCCESSFULLY!")
                else:
                    self.draw_text("YOU CRASHED!")

                if self.sp.check_orientation() \
                        and self.sp.check_land_spot() \
                        and self.sp.check_speed():
                    self.draw_text("YOU LANDED SUCCESSFULLY!")
                else:
                    self.draw_text("YOU CRASHED!")
                '''
            # Not yet update the message on the screen
            else:
                '''
                self.draw_text(
                    "Orient:{}  Land:{} speed:{}".format(
                        self.sp.check_orientation(),
                        self.sp.check_land_spot(),
                        self.sp.check_speed()
                    ))
                '''

            pygame.display.flip()
            self.screen.blit( self.background, (0, 0) )

        pygame.quit()

    def updateWeights(self):
        scores = np.zeros(config['num_ships'])
        for i in range(config['num_ships']):
            scores[i] = self.ships[i].fitness
        # Sort the scores from low value to high values
        # Low values indicate a better score (Closer to landing zone)
        scores_sort = scores.argsort()

        newShips = []

        #Take best performing ships(Top 30%) and introduce directly to next round
        num_bestShips = int(np.floor(config['num_ships']*0.3))
        for i in range(num_bestShips):
            newShips.append(self.ships[scores_sort[i]])

        #Mutate top 10% of ships 2 times, then reintroduce
        num_bestShips = int(np.floor(config['num_ships']*0.1))
        for i in range(num_bestShips):
            """We are going to combine all the weights into one 1D array.
            After chaning the weights, we need to reshape them back into their original form."""
            #The MLP Neural network for this ship
            NN = self.ships[scores_sort[i]].mlp

            #Store shape information for reconstruction
            s1 = NN.intercepts_[0]
            s2 = NN.coefs_[0].shape
            s3 = NN.coefs_[1].shape

            #Combine all weights into one array
            intercepts= np.concatenate( (NN.intercepts_[0],NN.intercepts_[1]))
            weights1 = NN.coefs_[0].flatten()
            weights2 = NN.coefs_[1].flatten()
            allWeights = np.concatenate((intercepts,weights1,weights2))

            #Mutate anywhere from 10% to %90 (need to multiply by 0.64 to get actual)
            num_m = int((np.random.rand()*0.8+0.1)*len(allWeights))
            #Array of indices to mutate (where 0.64 comes from)
            m_inds = np.random.randint(0,len(allWeights),num_m)

            for i in range(len(m_inds)):
                allWeights[m_inds[i]] = np.random.rand()*2-1

            #Reconstruct
            #intrcpts = allWeights[range(len(intercepts))]
            #coefs_0 = intrcpts[NN.intercepts_[0]]
            #self.ships[scores_sort[i]].mlp.coefs_[0] =

            #


            ##weights = np.zeros(1 + self.ships[scores_sort[i]].mlp.coefs_[0] + self.ships[scores_sort[i]].mlp.coefs_[1])
            ##weights[0] = self.ships[scores_sort[i]].mlp.bias
            ##weights[range(1,)]

        #Mutate second best ship 1 time




        if(self.stop_printing == False):
            print(scores)
            print(scores_sort)
        self.stop_printing = True
            #self.ships[i].mlp.coefs_



    def draw_text( self, text ):
        """
        """
        fw, fh = self.font.size(text)  # fw: font width,  fh: font height
        surface = self.font.render( text, True, (0, 255, 0) )
        # // makes integer division in python3
        self.screen.blit(
            surface, ( ( self.width - fw ), ( self.height - fh )) )


    def do_planet(self, radius, center=(200, 200) ):
        """Draw the planet including the gaussian noise
        to simulate erruptions"""

        # angle in degrees between points defining the planet
        res = 0.01*180/np.pi

        # numer of points defining the planet
        npoints = int( 360.0//res + 1)
        thetas = np.arange(0, 360, res)
        plist = np.zeros((npoints, 2))

        # the landing part of the planet
        fas = config["flat_angle_start"]
        faw = config["flat_angle_width"]

        # create a list of indexes were theta is in the flat range
        flat_thetas_index = []
        for ii in range(len(thetas)):
            if ( angle_between(thetas[ii], fas, faw) ):
               flat_thetas_index.append(ii)

        # Add gaussian noise of 2 pixels
        landform = np.random.normal( scale=2, size=( npoints, 2) )

        landform[ flat_thetas_index, : ] = 0

        plist[:, 0] = center[0] + radius*np.cos( thetas*np.pi/180 )
        plist[:, 1] = center[1] + radius*np.sin( thetas*np.pi/180 )

        pygame.draw.polygon( self.screen, colors.white, plist + landform )
        return plist[ [flat_thetas_index], : ]

    def resetShipLocs(self):
        """Reset the ship locations, but not their neural net weights"""

        for i in range(config['num_ships']):
            self.ships[i].pos = VEC((150, 150))
            self.ships[i].angle = 90
            self.ships[i].velocity = VEC(0, 0)
            self.ships[i].crashed = False
            self.ships[i].fitness = 0

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


        # reference vector for measuring angles.
        self.reference_vector = VEC(0, 1)

        # VEC can't be instantiated with array
        # so we convert to list

        self.mlp = MLPClassifier(hidden_layer_sizes=(4),max_iter=1)
        self.mlp.fit(X_train,y_train)
        #Initialize the MLP with random weights
        self.mlp.coefs_[0] = np.random.rand(3,4)*2-1
        self.mlp.coefs_[1] = np.random.rand(4,4)*2-1


    def predict(self):

        #########Calculate Inputs for neural network##########
        ship_coors = self.tip
        land_coors = self.landing_points[0]
        ship_angle = self.angle%360
        dSurface = (ship_coors - config["planet_center"]).length() - config["planet_radius"]
        dLandStrip = (ship_coors - land_coors).length()

        ########Update the ships fitness value###############
        self.fitness = dLandStrip

        #########Make prediction based on inputs##########
        string_output = "none"
        X = np.array([ship_angle,dSurface,dLandStrip])
        output = self.mlp.predict(X.reshape(1,-1))

        if(output==0):
            string_output = "none"
        elif(output==1):
            string_output = "left"
        elif(output==2):
            string_output = "right"
        elif(output==3):
            string_output = "up"

        return string_output

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
        # polar coordinates with planet at origin
        planet_distance = (self.pos - config["planet_center"])
        planet_angle = angle_to(planet_distance)

        fas = config["flat_angle_start"]
        faw = config["flat_angle_width"]

        return angle_betwen(planet_angle, fas, faw )

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

        return ship_angle,dSurface,dLandStrip



# End win condition methods.
if __name__ == '__main__':

    # call with width of window and fps
    PygView(1000, 800).run()

