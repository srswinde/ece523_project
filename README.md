

### ANN's Solving a lunar lander like game.


##### Configuration

There are several cofiguration settings located at the top of the program. These are in the config variable. You can control

-  The number of ships (num_ships)
-  Which ship to start with (previously trained or random). If you set load_ships to false the ship will be random otherwise the name of the pickled ship file must be set as the ship_file
-  The level or levels that are played in the "theLevels" variable. This is a list of level file names.
-  Whether or not you want to normalize the fitness across several levels with the "NormalizeFitness" If "theLevels" list only contains one level you probably want to set this to False

#### Running the Game

To run the game with the ANN playing simply run lunarlander.py with python3.


#### Requirements
 - Python 3.0+
 - Numpy
 - PyGame
 - pathlib
 - SciKit-Learn
 - Pickle

## To start the game type python3 lunarlander.py
## Rules
Simply navigate the triangle space ship to the
section of the planet that is not errupting.
The tip of the space ship must be pointing
away from the planet (orientation) the speed
must be below a certain threshold and you
must land on the correct part of the planet

## Keys

UP is thrust
Left is rotate CCW
Right is rotate CW
