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
    delta_angle=1,
    thrust=0.01,
    dt=2, #0.05
    flat_index = 0,
    num_ships = 30,
    planet_center = VEC( 500, 200 ),
    planet_center2 = VEC( 100, 100 ),
    speed_multiplier = 1.35,
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
        
        self.loadShips()

    def loadShips(self):
        with open('goodShips.pkl', 'rb') as f:
            lShipData = pickle.load(f)
        
        for i in range(config['num_ships']):
            self.ships[i].mlp.intercepts_[0] = lShipData[-1]['intercepts1']
            self.ships[i].mlp.intercepts_[1] = lShipData[-1]['intercepts2']
            self.ships[i].mlp.coefs_[0] = lShipData[-1]['weights1']
            self.ships[i].mlp.coefs_[1] = lShipData[-1]['weights2']

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
        while running:
            da = 0
            thrust = 0.0
            #initialize ship
            #self.ship = self.ships[count]
            
            for j in range(config['num_ships']):
            self.ships[j].physics(
                delta_angle=da,
                thrust=thrust,
                stop=self.ships[j].crashed)
            start_time = time.time() 
            all_crashed = False
            while all_crashed == False:
                self.draw_text("Generation:{}".format(self.generation))
                # Render the planet
                self.do_planet(
                    radius=config["planet_radius"],
                    center=config['planet_center'],
                    flat_index = config['flat_index'])

                for j in range(config['num_ships']):
                    if(time.time()-start_time > config['time_limit']):
                        self.ships[j].crashed = True
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
                    ai_key = self.ships[j].predict()
                    if ai_key == "left":
                        da = -config["delta_angle"]
                    if ai_key == "right":
                        da = config["delta_angle"]
                    thrust = config["thrust"]
                    # Do the physics on the spaceship
                    self.ships[j].physics(
                        delta_angle=da,
                        thrust=thrust,
                        stop=self.ships[j].crashed )
                    # Did we land?
                    if(self.ships[j].check_on_planet() or self.ships[j].check_pos_screen()==False):
                        self.ships[j].crashed = True

                    #Run this to update fitness
                    self.ships[j].updateFitness()

                pygame.display.flip()
                self.screen.blit( self.background, (0, 0) )

                all_crashed = True
                for j in range(config['num_ships']):
                    if(self.ships[j].crashed == False):
                        all_crashed = False


            #self.updateWeights()
            self.resetShipLocs()
        pygame.quit()

    def mutate(self,mlp):
        """We are going to combine all the weights into one 1D array.
        After chaning the weights, we need to reshape them back into their original form."""
        #The MLP Neural network for this ship
        NN = deepcopy(mlp)

        #Store shape information for reconstruction
        s0 = len(NN.intercepts_[0])
        s1 = len(NN.intercepts_[1])
        s2 = NN.coefs_[0].shape
        s3 = NN.coefs_[1].shape

        #Combine all weights into one array
        intercepts= np.concatenate( (NN.intercepts_[0],NN.intercepts_[1]))
        weights1 = NN.coefs_[0].flatten()
        weights2 = NN.coefs_[1].flatten()
        allWeights = np.concatenate((weights1,weights2))

        #Mutate anywhere from 5% to %20
        num_m_weights = 1;#int((np.random.rand()*0.10+0.05)*len(allWeights))
        num_m_intercepts = int(np.round(np.random.rand()));# int((np.random.rand()*0.10+0.05)*len(intercepts))

        #Array of indices to mutate
        m_inds_w = np.random.choice(range(0,len(allWeights)), size = num_m_weights, replace = False)
        m_inds_i = np.random.choice(range(0,len(intercepts)), size = num_m_intercepts, replace = False)


        selector = np.random.rand()
        if(selector > 0.5):

            mutateFactor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
            for ii in range(len(m_inds_w)):
                allWeights[m_inds_w[ii]] = allWeights[m_inds_w[ii]] * mutateFactor

            mutateFactor = 1 + ((np.random.rand() - 0.5) * 3 + (np.random.rand() - 0.5))
            if(num_m_intercepts!=0):
                for ii in range(len(m_inds_i)):
                    intercepts[m_inds_i[ii]] = allWeights[m_inds_w[ii]] * mutateFactor
        else:

            for ii in range(len(m_inds_w)):
                allWeights[m_inds_w[ii]] = np.random.rand()*2-1

            if(num_m_intercepts!=0):
                for ii in range(len(m_inds_i)):
                    intercepts[m_inds_i[ii]] = np.random.rand()*2-1



        #Reconstruct
        intercepts_0 = intercepts[range(s0)]
        intercepts_1 = intercepts[range(s0,s1+s0)]
        coefs_0 = allWeights[range(len(weights1))].reshape(s2)
        coefs_1 = allWeights[range(len(weights1),len(weights2)+len(weights1))].reshape(s3)

        #Add the new weights back into the neural network
        NN.intercepts_[0] = intercepts_0
        NN.intercepts_[1] = intercepts_1
        NN.coefs_[0] = coefs_0
        NN.coefs_[1] = coefs_1

        return deepcopy(NN)

    def crossover(self,mlp1,mlp2):
        """We are going to combine all the weights into one 1D array.
        After chaning the weights, we need to reshape them back into their original form."""
        #The MLP Neural network for this ship
        NN = deepcopy(mlp1)
        NN2 = deepcopy(mlp2)

        #Store shape information for reconstruction
        s0 = len(NN.intercepts_[0])
        s1 = len(NN.intercepts_[1])
        s2 = NN.coefs_[0].shape
        s3 = NN.coefs_[1].shape

        intercepts= np.concatenate( (NN.intercepts_[0],NN.intercepts_[1]))
        weights1 = NN.coefs_[0].flatten()
        weights2 = NN.coefs_[1].flatten()
        allWeights = np.concatenate((weights1,weights2))

        intercepts2= np.concatenate( (NN2.intercepts_[0],NN2.intercepts_[1]))
        weights12 = NN2.coefs_[0].flatten()
        weights22 = NN2.coefs_[1].flatten()
        allWeights2 = np.concatenate((weights12,weights22))

        #Crossover anywhere from 20% to %60
        #Number of weights and intercepts to crossover
        num_m_weights = 1 #int( np.ceil( (np.random.rand()*0.1)*len(allWeights)) )
        num_m_intercepts = int(np.round(np.random.rand())); #np.round((np.random.rand()*0.1)*len(intercepts))

        m_inds_w = np.random.choice(range(0,len(allWeights)), size = num_m_weights, replace = False)
        m_inds_i = np.random.choice(range(0,len(intercepts)), size = num_m_intercepts, replace = False)

        for ii in range(len(m_inds_w)):
            allWeights[m_inds_w[ii]] = allWeights2[m_inds_w[ii]]

        if(num_m_intercepts !=0):
            for ii in range(len(m_inds_i)):
                intercepts[m_inds_i[ii]] = intercepts2[m_inds_i[ii]]

        #Reconstruct
        intercepts_0 = intercepts[range(s0)]
        intercepts_1 = intercepts[range(s0,s1+s0)]
        coefs_0 = allWeights[range(len(weights1))].reshape(s2)
        coefs_1 = allWeights[range(len(weights1),len(weights2)+len(weights1))].reshape(s3)

        #Add the new weights back into the neural network
        NN.intercepts_[0] = intercepts_0
        NN.intercepts_[1] = intercepts_1
        NN.coefs_[0] = coefs_0
        NN.coefs_[1] = coefs_1

        return deepcopy(NN)


    def updateWeights(self):
        newShips = []
        
        scores = np.zeros(config['num_ships'])
        for i in range(config['num_ships']):
            scores[i] = deepcopy(self.ships[i].fitness)

        scores_sort = np.sort(scores)
        #Invert. Make highest scores to Lowest
        scores_sort = 1/scores_sort

        if(scores_sort[0]> self.bestScore):
            self.bestScore = scores_sort[0]
        scores_sort_ind = scores.argsort()

        ##### PRINT STUFF #####
        print("")
        print("Generation: " , self.generation)


        #If we did worse than before, reject this generation
        reject = False
        if(scores_sort[0]<self.bestScore):
            scores = np.zeros(config['num_ships'])
            for i in range(config['num_ships']):
                scores[i] = deepcopy(self.prevFitness[i])
                self.ships[i].mlp = deepcopy(self.prevShips[i])
                self.ships[i].fitness = deepcopy(self.prevFitness[i])

            scores_sort = np.sort(scores)
            #Invert. Make highest scores to Lowest
            scores_sort = 1/scores_sort
            print("Generation Rejected")
            reject = True
        
        self.generation = self.generation + 1
        for i in range(config['num_ships']):
            #Get Weight value of best ship
            NN1= deepcopy(self.ships[scores_sort_ind[i]].mlp)
            intercepts= np.concatenate( (NN1.intercepts_[0],NN1.intercepts_[1]))
            weights1 = NN1.coefs_[0].flatten()
            weights2 = NN1.coefs_[1].flatten()
            allWeights = np.concatenate((intercepts,weights1,weights2))
            weightSum = deepcopy(np.sum(allWeights))


            # pickle info of best ship
            if scores_sort_ind[i] == 0 and not reject:
                logdict = {
                    'ship_num': i,
                    'weights1': NN1.coefs_[0],
                    'weights2': NN1.coefs_[1],
                    'intercepts1': NN1.intercepts_[0],
                    'intercepts2': NN1.intercepts_[1],
                    'Generation': self.generation,
                    'timestamp': datetime.datetime.now(),
                    'score':self.ships[scores_sort_ind[i]].fitness
                }
                self.logLst.append(logdict)
                fname = "best.pkl"
                #fname = "{}_best.pkl".format(datetime.datetime.now().isoformat().replace(':','-'))

                with open(fname, 'wb') as pfd:
                    pickle.dump(  self.logLst, pfd )

            print("Ship Score:",self.ships[scores_sort_ind[i]].fitness,"Weight:" , weightSum)
        #print(self.bestScore)
        #########################


        # Sort the scores from low value to high values
        # Low values indicate a better score (Closer to landing zone)
        scores_sort_ind = scores.argsort()
        sortedShips = []
        for i in range(config['num_ships']):
            sortedShips.append( deepcopy(self.ships[scores_sort_ind[i]].mlp))

       #Normalize the fitness scores
        scores_sum = np.sum(scores_sort)
        scores_sort = scores_sort/scores_sum
        probabilities = scores_sort

        #Take best performing ships(Top 20%) and introduce directly to next round
        num_bestShips = int(np.floor(config['num_ships']*0.2))
        for i in range(num_bestShips):
            newShips.append(deepcopy(self.ships[scores_sort_ind[i]].mlp))

        #Take two parents, mutate them, and introduce to next round (Skip crossover)
        for i in range(2):
            parents1 = np.random.choice(range(config['num_ships']),size = 2, replace = False,p=probabilities)
            theNewMlp1 = self.mutate(sortedShips[parents1[0]])
            newShips.append(deepcopy(theNewMlp1))

        #Whatever ships we have left mutate + crossbreed
        for i in range(int(config['num_ships'] - len(newShips))):
            #Select two parents
            parents = np.random.choice(range(config['num_ships']),size = 2, replace = False,p=probabilities)

            NN = self.crossover(sortedShips[parents[0]],sortedShips[parents[1]])
            theNewMlp = self.mutate(NN)
            #theNewMlp = self.mutate(sortedShips[parents[0]])

            newShips.append(deepcopy(theNewMlp))

   

        #Save the previous ships incase all the new ships are worse
        self.prevShips = []
        self.prevFitness = []
        for i in range(len(self.ships)):
            self.prevShips.append( deepcopy(self.ships[i].mlp))
            self.prevFitness.append( deepcopy(self.ships[i].fitness))
            self.ships[i].mlp = deepcopy(newShips[i])
    

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

    def updateFitness(self):
        ########Update the ships fitness value###############
        # Fitness is defined as the distance from the landing strip, dLandStrip
        self.fitness = self.inputs[2]

    def predict(self):

        #########Calculate Inputs for neural network##########
        ship_coors = self.pos
        land_coors = self.landing_points[0]
        ship_angle = self.angle%360
        dSurface = (ship_coors - config["planet_center"]).length() - config["planet_radius"]
        dLandStrip = (ship_coors - self.mid_landing_point).length()

        #########Normalize inputs, want 0 to 1 range########

        ship_angle = ship_angle/360
        maxD = VEC(1000,800).length()
        dSurface = dSurface/maxD
        dLandStrip = dLandStrip/maxD
        minDLandStrip = self.minDLandStrip/maxD

        #########Make prediction based on inputs##########
        string_output = "none"
        X = np.array([ship_angle,dSurface,dLandStrip])
        self.inputs = X 
        output = self.mlp.predict(X.reshape(1,-1))[0]

        if(output==0):
            string_output = "left"
        elif(output==1):
            string_output = "right"
        return string_output


    def render(self, color ):

        tip = VEC( 10, 0)
        left = VEC(-5, 5)
        right = VEC(-5, -5)

        for pt in (tip, right, left):
            pt.rotate_ip( self.angle )
            pt += self.pos
        pygame.draw.polygon(self.screen, color, ( tip, left, right ) )

        self.tip, self.left, self.right = tip, left, right

    def physics( self, thrust=0.0, delta_angle=0.0, stop=False ):
        ppos =  config["planet_center"]

        gravity = config["gravity"]*(self.pos-ppos).normalize()
        dt = config["dt"]
        if not stop:
            thrust_vector = VEC(1, 0).rotate(self.angle)*thrust
            # self.velocity = self.velocity + (gravity+thrust_vector)*dt
            self.velocity = config['speed_multiplier']*VEC(1, 0).rotate(self.angle)

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
        