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
    blue = (0,142,204)
    black = (0, 0, 0)


LOCS = [VEC( 800, 600 ),VEC( 350, 450 ),VEC( 300, 750 )]

mode = "hard"

if(mode == 'easy'):
    radii_red = [130,130,130]#[100,75,130,50]
    centers_red = [(250,250),(250,600),(600,250)]#[(150,250),(100,300),(800,200),(350,200)]
    center_white = VEC(800,600)
elif(mode == 'medium'):
    radii_red = [130,130,150]#[100,75,130,50]
    centers_red = [(350,250),(300,700),(650,300)]#[(150,250),(100,300),(800,200),(350,200)]
    center_white = VEC(800,600)
elif(mode == 'hard'):
    radii_red = [130,130,150,100,100,100]#[100,75,130,50]
    centers_red = [(350,250),(300,700),(650,300),(800,100),(520,560),(800,560)]#[(150,250),(100,300),(800,200),(350,200)]
    center_white = VEC(1000,130)



config = dict(
    planet_radius=45,
    gravity= 0,  #-0.002,
    land_angle=10,
    land_speed=0.25,
    delta_angle=2,
    thrust=0.01,
    dt=5, #0.05
    flat_index = 0,
    num_ships = 10,
    starting_pos = (20,20),
    starting_angle = 45,
    planet_center = center_white, #VEC( 300, 750 ),#VEC( 800, 600 ),
    planet_center2 = VEC( 100, 100 ),
    speed_multiplier = 1.35,

    # If num_planets > 1 each
    # extra planet will be a "bad" planet
    num_planets = 2,
    red_planet_size = 100,
    random_planets = False,
    time_limit = 30,
    load_ships = True

)


#Neural Network Structure
n_inputs = 10
n_hidden = 7
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
        self.planetFinished = False
        #config["planet_center"] = VEC( self.width//2, self.height//2 )
        self.planets = []
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

        if(config['load_ships']==True):
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
                    ai_key = self.ships[j].predict(self.planets)
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

                    self.ships[j].updateFitness(config['planet_center'])

                    if ( self.ships[j].check_red_planets(self.planets) == False ):
                        self.ships[j].crashed = True
                        # Give it a mean Penalty.
                        self.ships[j].fitness = self.ships[j].fitness + 0.2

                    #Run this again to update fitness
                    #_ = self.ships[j].predict()
                    #Run this to update fitness
                    

                pygame.display.flip()
                self.screen.blit( self.background, (0, 0) )

                all_crashed = True
                for j in range(config['num_ships']):
                    if(self.ships[j].crashed == False):
                        all_crashed = False


            self.updateWeights()
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
        num_m_weights = int(np.round((np.random.rand()*0.15+0.05) * len(allWeights))) #int((np.random.rand()*0.10+0.05)*len(allWeights))
        num_m_intercepts = int(np.round((np.random.rand()*0.15+0.05) * len(intercepts))) * int(np.round(np.random.rand()))#int(np.round(np.random.rand()));# int((np.random.rand()*0.10+0.05)*len(intercepts))

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
        #num_m_weights = 3 #int( np.ceil( (np.random.rand()*0.1)*len(allWeights)) )
        #num_m_intercepts = int(np.round(np.random.rand())); #np.round((np.random.rand()*0.1)*len(intercepts))
        num_m_weights = int(np.round((np.random.rand()*0.15+0.05)  * len(allWeights))) #int((np.random.rand()*0.10+0.05)*len(allWeights))
        num_m_intercepts = int(np.round((np.random.rand()*0.15+0.05)  * len(intercepts))) * int(np.round(np.random.rand()+0.3))

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
            scores[i] = deepcopy(self.ships[i].fitness2)

        scores_sort = np.sort(scores)[::-1]
        #Invert. Make highest scores to Lowest
        #scores_sort = 1/scores_sort

        if(scores_sort[0]> self.bestScore):
            self.bestScore = scores_sort[0]
        scores_sort_ind = scores.argsort()[::-1] #Descending order (highest to lowest)

        ##### PRINT STUFF #####
        print("")
        print("Generation: " , self.generation)


        #If we did worse than before, reject this generation
        
        reject = False
        '''
        if(scores_sort[0]<self.bestScore):
            scores = np.zeros(config['num_ships'])
            for i in range(config['num_ships']):
                scores[i] = deepcopy(self.prevFitness[i])
                self.ships[i].mlp = deepcopy(self.prevShips[i])
                self.ships[i].fitness2 = deepcopy(self.prevFitness[i])

            scores_sort = np.sort(scores)[::-1]
            #Invert. Make highest scores to Lowest
            #scores_sort = 1/scores_sort
            print("Generation Rejected")
            reject = True
        '''
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
                    'score':self.ships[scores_sort_ind[i]].fitness2
                }
                self.logLst.append(logdict)
                fname = "best.pkl"
                #fname = "{}_best.pkl".format(datetime.datetime.now().isoformat().replace(':','-'))

                with open(fname, 'wb') as pfd:
                    pickle.dump(  self.logLst, pfd )

            print("Ship Score:",self.ships[scores_sort_ind[i]].fitness2,self.ships[scores_sort_ind[i]].fitnessDebug, "Weight:" , weightSum)
        #print(self.bestScore)
        #########################


        # Sort the scores from low value to high values
        # Low values indicate a better score (Closer to landing zone)
        scores_sort_ind = scores.argsort()[::-1]
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
            self.prevFitness.append( deepcopy(self.ships[i].fitness2))
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

        
        if config['random_planets'] == True:
            new_planet_angle = 180
            if config['num_planets'] > 1:
                for pl in range( config['num_planets']+1 ):
                    if len( self.planets ) != config['num_planets']+1:
                        npx0 = np.random.randint(250, 600)

                        np_center = VEC(center) + VEC(npx0, 0).rotate(new_planet_angle)

                        self.planets.append((np_center, config['red_planet_size']))
                        new_planet_angle+=30
                    plcenter, plradius = self.planets[pl]
                    pygame.draw.circle(self.screen, colors.red, np.int64(plcenter), plradius )
                
        else:
            """Use our pre-programmed course"""
            radii = radii_red      #[130,90,150,100]#[100,75,130,50]
            centers = centers_red  #[(350,250),(300,600),(650,300),(800,100)]#[(150,250),(100,300),(800,200),(350,200)]
            for i in range(len(centers)): 
                np_center = VEC(centers[i]) 
                if self.planetFinished == False:
                    self.planets.append((np_center, radii[i]))
                plcenter, plradius = self.planets[i]
                pygame.draw.circle(self.screen, colors.red, np.int64(plcenter), plradius )   
            self.planetFinished = True
                       


        return plist[ fi0:fi1, : ]

    def resetShipLocs(self):
        """Reset the ship locations, but not their neural net weights"""

        for i in range(config['num_ships']):
            self.ships[i].pos = config['starting_pos']
            self.ships[i].angle = config['starting_angle']
            self.ships[i].velocity = VEC(0, 0)
            self.ships[i].crashed = False
            self.ships[i].fitness2 = 0
            self.ships[i].fitnessDebug = 0
            self.ships[i].sawTheGoodPlanet = False
            self.ships[i].donezo = False

class space_ship:
    """The space shipe class"""
    def __init__(self, screen, landing_points, pos=(50, 30), angle=90 ):
        self.pos = config['starting_pos']
        self.angle = config['starting_angle']
        self.screen = screen
        self.velocity = VEC(0, 0)
        self.landing_points = landing_points
        self.crashed = False
        self.fitness = 0
        self.inputs = np.zeros(n_inputs)
        self.fitness2 = 0 
        self.fitnessDebug = 0
        # find mid point of landing
        li = landing_points.shape[0]//2
        self.mid_landing_point = VEC(list(self.landing_points[li]))
        self.sawTheGoodPlanet = False
        self.donezo = False

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

    def updateFitness(self,planetCenter):
        ########Update the ships fitness value###############
        # Fitness is defined as the distance from the landing strip, dLandStrip
        maxD = VEC(1000,800).length()
        bad_distances = 0
        good_distances = 0
        badCount = 0 
        goodCount = 0

        distances = self.inputs[range(5)]
        bad = self.inputs[range(5,10)]

        bad_inds = np.where(bad == 1)[0]
        bad_distances = distances[bad_inds]
        bad_distances = np.min(bad_distances)

        
        good_inds = np.where(bad == 0)[0]
        if(len(good_inds) !=0):
            good_distances = distances[good_inds]
            good_distances = np.min(good_distances)
            good_distances = 1/good_distances
            self.fitnessDebug = self.fitnessDebug + good_distances * maxD*10
            self.fitness2 = self.fitness2 + bad_distances + good_distances * maxD*10
            self.sawTheGoodPlanet = True
        else:
            self.fitness2 = self.fitness2 + bad_distances


        #If we see the planet (once) double my current fitness score. 
        #Encourages ships to come into view of the good planet
        if(self.sawTheGoodPlanet == True and self.donezo == False):
            self.fitness2 = self.fitness2 * 2
            self.donezo = True

        #########Calculate Inputs for fitness##########
        #ship_coors = self.pos
        #dPlanet = (ship_coors - planetCenter).length()

        #########Normalize inputs, want 0 to 1 range########
        #maxD = VEC(1000,800).length()
        #dPlanet = dPlanet/maxD

        #self.fitness = dPlanet

    def predict(self,red_planets):
        string_output = "none"
        X = self.calcInputs(red_planets)
        
        #########Normalize inputs, want 0 to 1 range########
        self.inputs = deepcopy(X)
        maxD = VEC(1000,800).length()
        X[range(5)] = X[range(5)]/maxD

        #########Make prediction based on inputs##########
        output = self.mlp.predict(X.reshape(1,-1))[0]
        if(output==0):
            string_output = "left"
        elif(output==1):
            string_output = "right"
        return string_output
    
    def calcInputs(self,red_planets):
        avoidObject = np.zeros(5)
        objectDistances = np.zeros(5)
        #For each direction
        for i in range(5):
            #For each planet (+1 is for the good planet)
            allObjDistances = []
            for j in range(len(red_planets)+1):
                distFromEdge = self.wallIntercept(i)
                #avoidObject[i] = 1
                if(j!=len(red_planets)):#If we're not equal to the last planet (that's the good one)
                    #red_planets[j][0] is the planet center
                    dist = self.circleIntercept(i,red_planets[j][0],red_planets[j][1])
                    if(dist == -1):
                        dist = distFromEdge
                    allObjDistances.append(dist)
                else:
                    #Make the last planet the good one
                    center = np.array([config['planet_center'][0],config['planet_center'][1]])
                    dist = self.circleIntercept(i,center,config['planet_radius'])
                    if(dist == -1):
                        dist = 99999
                    allObjDistances.append(dist)
            objectDistances[i] = min(allObjDistances)
            ind = allObjDistances.index(objectDistances[i])
            if(ind != len(red_planets)):
                avoidObject[i] = 1

        return np.concatenate((objectDistances,avoidObject))


    def wallIntercept(self,direction):
            #m is the slope of the line. Used to describe line in direction of ship
            #direction = 4
            
            if(direction ==0):
                #straight
                m = self.tip - self.back  
                x, y = self.tip[0],self.tip[1]          
            if(direction == 1):
                #left
                m = self.left - self.right
                x, y = self.left[0],self.left[1]        
            if(direction == 2):
                #right
                m = self.right - self.left
                x, y = self.right[0],self.right[1]       
            if(direction == 3):
                #left-staight
                m = (self.left + self.tip)/2 - self.right 
                x, y = ((self.left + self.tip)/2)[0],((self.left + self.tip)/2)[1]
            if(direction == 4):
                #right-straight
                m = (self.right + self.tip)/2 - self.left 
                x, y = ((self.right + self.tip)/2)[0],((self.right + self.tip)/2)[1]
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
                        if (self.back - VEC(x_i,y_i)).length() < (self.tip - VEC(x_i,y_i)).length():
                            dist = -1   
                    if(direction == 1):
                        #left
                        if (self.right - VEC(x_i,y_i)).length() < (self.left - VEC(x_i,y_i)).length():
                            dist = -1    
                    if(direction == 2):
                        #right
                        if (self.left - VEC(x_i,y_i)).length() < (self.right - VEC(x_i,y_i)).length():
                            dist = -1      
                    if(direction == 3):
                        #left-staight
                        if (self.right - VEC(x_i,y_i)).length() < ((self.left + self.tip)/2 - VEC(x_i,y_i)).length():
                            dist = -1  
                    if(direction == 4):
                        #right-straight
                        if (self.left - VEC(x_i,y_i)).length() < ((self.right + self.tip)/2 - VEC(x_i,y_i)).length():
                            dist = -1                   
                if(dist != -1):
                    lDistances.append(dist)
            
            #For some reason, it didn't get any distances once. This will prevent the game from crashing if that happens
            if len(lDistances) == 0:
                lDistances.append(1)
                #print("Bug!")
            return np.min(lDistances)

    def circleIntercept(self,direction,planetCenter,planetRadius):
        """https://math.stackexchange.com/questions/228841/how-do-i-calculate-the-intersections-of-a-straight-line-and-a-circle"""
        
        #m is the slope of the line. c is the y intercept. used to describe line in direction of ship
        #direction = 4
        
        if(direction == 0):
            #straight
            m = self.tip - self.back  
            lineStart = self.tip          
        if(direction == 1):
            #left
            m = self.left - self.right
            lineStart = self.left       
        if(direction == 2):
            #right
            m = self.right - self.left     
            lineStart = self.right 
        if(direction == 3):
            #left-staight
            m = (self.left + self.tip)/2 - self.right 
            lineStart = (self.left + self.tip)/2
        if(direction == 4):
            #right-straight
            m = (self.right + self.tip)/2 - self.left 
            lineStart = (self.right + self.tip)/2
        #Don't want to divide by zero, so just give m a really high value if x in y/x is 0
        if(m[0]==0):
            m=999999
        else:
            m = m[1]/m[0]     

        #We want left and right 'seeing directions' to be at the back of the ship
        c = lineStart[1] - m * lineStart[0]


        p = planetCenter[0]  #config['planet_center'][0]
        q = planetCenter[1]  #config['planet_center'][1]
        r = planetRadius #config['planet_radius']

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
                if (self.back - VEC(x,y)).length() < (self.tip - VEC(x,y)).length():
                    dist = -1          
            if(direction == 1):
                #left
                if (self.right - VEC(x,y)).length() < (self.left - VEC(x,y)).length():
                    dist = -1      
            if(direction == 2):
                #right
                if (self.left - VEC(x,y)).length() < (self.right - VEC(x,y)).length():
                    dist = -1      
            if(direction == 3):
                #left-staight
                if (self.right - VEC(x,y)).length() < ((self.left + self.tip)/2 - VEC(x,y)).length():
                    dist = -1  
            if(direction == 4):
                #right-straight
                if (self.left - VEC(x,y)).length() < ((self.right + self.tip)/2 - VEC(x,y)).length():
                    dist = -1             
        return dist

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

        # gravity = config["gravity"]*(self.pos-ppos).normalize()
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
            color = colors.blue

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

    def check_red_planets(self, rps):
        for ppos, rad in rps:
            if (self.tip - ppos).length() < rad:
                return False

        return True



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





# End win condition methods.
if __name__ == '__main__':

    # call with width of window and fps
    PygView(1000, 800).run()
    #If we want to run the game without rendering to the screen
    PygView(1000, 800).run()

