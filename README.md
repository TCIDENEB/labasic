Week 1 + 2






Task #1: Write a for loop to print all the numbers from 1 to 10.

Code:
i = 1
while i <= 10:
 print(i)
 i += 1




Task #2: Write a for loop to print the squares of numbers from 1 to 5.

Code:
i = 1
while i <= 5:
 print(i ** 2)
 i += 1




Task #3: Write a for loop to calculate the sum of all numbers from 1 to 100.

Code:
total_sum = 0
i = 1
while i <= 100:
 total_sum += i
 i += 1
print("Sum of numbers from 1 to 100:", total_sum)







Week 3















Task #1: Write a python program that searches a number in a list consisting of 10 elements.

Code:
array = [5, 2, 8, 10, 15, 3, 7, 12, 9, 6]
target = 10
for num in array:
 if num == target:
  print("Number 10 found in the list.")
  break
else:
  print("Number 10 not found in the list.")




Task #2: Write a program to find the maximum value in a list of
numbers.

Code:
numbers = [5, 8, 2, 10, 6, 3]
max_value = max(numbers)
print("Maximum value in the list:", max_value)




Task #3: Write a program to find the minimum value in a list of
numbers.

Code:
numbers = [5, 8, 2, 10, 6, 3]
min_value = min(numbers)
print("Minimum value in the list:", min_value)






Task #4: Write a program to find the index of a specific value in a
list.

Code:
def linear_search(arr, target):
 for i in range(len(arr)):
  if arr[i] == target:
   return i
 return -1
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
target_value = 5
result = linear_search(my_list, target_value)
if result != -1:
 print(f"Found {target_value} at index {result}")
else:
 print(f"{target_value} not found in the list")




Task #5: Write a program to count the number of occurrences of a
specific value in a list.

Code:
def linear_search(arr, target):
 x = 0
 for i in range(len(arr)):
  if arr[i] == target:
   x+=1
 return x
my_list = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]
target_value = 7
result = linear_search(my_list, target_value)
if result != -1:
 print(f"Found {target_value} like {result} times")




Task #6: DFS code

Code:
graph = {
 '5' : ['3','7'],
 '3' : ['2', '4'],
 '7' : ['8'],
 '2' : [],
 '4' : ['8'],
 '8' : []
}
visited = set() # Set to keep track of visited nodes of graph.
def dfs(visited, graph, node): #function for dfs
 if node not in visited:
  print (node)
  visited.add(node)
  for neighbour in graph[node]:
   dfs(visited, graph, neighbour)
print("Following is the Depth-First Search")
dfs(visited, graph, '5')








Week 4



Video to frames




import cv2

def video_to_frames(input_video_path, output_frame_dir, frame_rate):
    # Open the video file
    cap = cv2.VideoCapture(input_video_path)

    # Get the frames per second (fps) of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Calculate the frame interval for the desired frame rate
    frame_interval = fps // frame_rate

    frame_count = 0

    # Create the output directory if it doesn't exist
    import os
    os.makedirs(output_frame_dir, exist_ok=True)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Save the frame
        if frame_count % frame_interval == 0:
            frame_filename = f"{output_frame_dir}/frame_{frame_count}.jpg"
            cv2.imwrite(frame_filename, frame)

        frame_count += 1

    cap.release()

if __name__ == "__main__":
    input_video_path = r"/content/drive/MyDrive/input/cars.mp4"  # Replace with your input video file path
    output_frame_dir = r"/content/drive/MyDrive/output/"  # Output directory to save frames
frame_rate = 5  # Desired frame rate in frames per second

    video_to_frames(input_video_path, output_frame_dir, frame_rate)










Week 5













Task #1: Write the algorithm for greedy best first search

Code:
def greedy_best_first_search(graph, start, goal, heuristic):
    visited = set()
    priority_queue = [(heuristic[start], start)]
    
    while priority_queue:
        h, current = priority_queue.pop(0)
        visited.add(current)
        
        if current == goal:
            print("Goal found!")
            return
        
        print("Current node:", current)
        for neighbour in graph[current]:
            if neighbour not in visited:
                priority_queue.append((heuristic[neighbour], neighbour))
                priority_queue.sort()  # Sort based on heuristic value

    print("Goal not found!")

# Example graph
graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D'],
    'C': ['A', 'D'],
    'D': ['B', 'C']
}

# Example heuristic values
heuristic_values = {
    'A': 7,
    'B': 5,
    'C': 3,
    'D': 1
}

# Perform search
greedy_best_first_search(graph, 'A', 'D', heuristic_values)










Week 6






A star code




def a_star_search(graph, start, goal, heuristic):
    visited = set()
    open_list = [(heuristic[start], start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        open_list.sort()  # Sort based on heuristic value
        _, current = open_list.pop(0)
        visited.add(current)

        if current == goal:
            print("Goal found!")
            return

        print("Current node:", current)
        for neighbour, cost in graph[current].items():
            new_cost = cost_so_far[current] + cost
            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                priority = new_cost + heuristic[neighbour]
                open_list.append((priority, neighbour))
                came_from[neighbour] = current

    print("Goal not found!")

# Example graph
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'D': 5},
    'C': {'A': 4, 'D': 3},
    'D': {'B': 5, 'C': 3}
}

# Example heuristic values
heuristic_values = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 0
}

# Perform search
a_star_search(graph, 'A', 'D', heuristic_values)







ANN

import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(train_images, train_labels) , (test_images, test_labels) = mnist.load_data()

# Printing the shapes
print("train_images shape: ", train_images.shape)
print("train_labels shape: ", train_labels.shape)
print("test_images shape: ", test_images.shape)
print("test_labels shape: ", test_labels.shape)

# Displaying first 9 images of dataset
fig = plt.figure(figsize=(10,10))

nrows=3
ncols=3
for i in range(9):
  fig.add_subplot(nrows, ncols, i+1)
  plt.imshow(train_images[i])
  plt.title("Digit: {}".format(train_labels[i]))
  plt.axis(False)
plt.show()

# Converting image pixel values to 0 - 1
train_images = train_images / 255
test_images = test_images / 255

print("First Label before conversion:")
print(train_labels[0])

# Converting labels to one-hot encoded vectors
train_labels = tf.keras.utils.to_categorical(train_labels)
test_labels = tf.keras.utils.to_categorical(test_labels)

print("First Label after conversion:")
print(train_labels[0])

# Defining Model
# Using Sequential() to build layers one after another
model = tf.keras.Sequential([

  # Flatten Layer that converts images to 1D array
  tf.keras.layers.Flatten(),

  # Hidden Layer with 512 units and relu activation
  tf.keras.layers.Dense(units=512, activation='relu'),

  # Output Layer with 10 units for 10 classes and softmax activation
  tf.keras.layers.Dense(units=10, activation='softmax')
])

model.compile(
  loss = 'categorical_crossentropy',
  optimizer = 'adam',
  metrics = ['accuracy']
)
 #Training a neural network
history = model.fit(
  x = train_images,
  y = train_labels,
  epochs = 3
)

# Showing plot for loss
plt.plot(history.history['loss'])
plt.xlabel('epochs')
plt.legend(['loss'])
plt.show()

# Showing plot for accuracy
plt.plot(history.history['accuracy'], color='orange')
plt.xlabel('epochs')
plt.legend(['accuracy'])
plt.show()

# Call evaluate to find the accuracy on test images
test_loss, test_accuracy = model.evaluate(
  x = test_images,
  y = test_labels
)

print("Test Loss: %.4f"%test_loss)
print("Test Accuracy: %.4f"%test_accuracy)

# Making Predictions
predicted_probabilities = model.predict(test_images)
predicted_classes = tf.argmax(predicted_probabilities, axis=-1).numpy()

index=11

# Showing image
plt.imshow(test_images[index])

# Printing Probabilities
print("Probabilities predicted for image at index", index)
print(predicted_probabilities[index])

print()

# Printing Predicted Class
print("Probabilities class for image at index", index)
print(predicted_classes[index])





A-star simple



def a_star_search(graph, start, goal, heuristic):
    visited = set()
    open_list = [(heuristic[start], start)]
    came_from = {}
    cost_so_far = {start: 0}

    while open_list:
        open_list.sort()  # Sort based on heuristic value
        _, current = open_list.pop(0)
        visited.add(current)

        if current == goal:
            print("Goal found!")
            return

        print("Current node:", current)
        for neighbour, cost in graph[current].items():
            new_cost = cost_so_far[current] + cost
            if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]:
                cost_so_far[neighbour] = new_cost
                priority = new_cost + heuristic[neighbour]
                open_list.append((priority, neighbour))
                came_from[neighbour] = current

    print("Goal not found!")

# Example graph
graph = {
    'A': {'B': 2, 'C': 4},
    'B': {'A': 2, 'D': 5},
    'C': {'A': 4, 'D': 3},
    'D': {'B': 5, 'C': 3}
}

# Example heuristic values
heuristic_values = {
    'A': 5,
    'B': 4,
    'C': 2,
    'D': 0
}

# Perform search
a_star_search(graph, 'A', 'D', heuristic_values)





A-star:






from __future__ import annotations
from typing import Protocol, Iterator, Tuple, TypeVar, Optional
T = TypeVar('T')

Location = TypeVar('Location')
class Graph(Protocol):
    def neighbors(self, id: Location) -> list[Location]: pass

class SimpleGraph:
    def __init__(self):
        self.edges: dict[Location, list[Location]] = {}
    
    def neighbors(self, id: Location) -> list[Location]:
        return self.edges[id]

example_graph = SimpleGraph()
example_graph.edges = {
    'A': ['B'],
    'B': ['C'],
    'C': ['B', 'D', 'F'],
    'D': ['C', 'E'],
    'E': ['F'],
    'F': [],
}

import collections

class Queue:
    def __init__(self):
        self.elements = collections.deque()
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, x: T):
        self.elements.append(x)
    
    def get(self) -> T:
        return self.elements.popleft()

# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)

def draw_tile(graph, id, style):
    r = " . "
    if 'number' in style and id in style['number']: r = " %-2d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = " > "
        if x2 == x1 - 1: r = " < "
        if y2 == y1 + 1: r = " v "
        if y2 == y1 - 1: r = " ^ "
    if 'path' in style and id in style['path']:   r = " @ "
    if 'start' in style and id == style['start']: r = " A "
    if 'goal' in style and id == style['goal']:   r = " Z "
    if id in graph.walls: r = "###"
    return r

def draw_grid(graph, **style):
    print("___" * graph.width)
    for y in range(graph.height):
        for x in range(graph.width):
            print("%s" % draw_tile(graph, (x, y), style), end="")
        print()
    print("~~~" * graph.width)

# data from main article
DIAGRAM1_WALLS = [from_id_width(id, width=30) for id in [21,22,51,52,81,82,93,94,111,112,123,124,133,134,141,142,153,154,163,164,171,172,173,174,175,183,184,193,194,201,202,203,204,205,213,214,223,224,243,244,253,254,273,274,283,284,303,304,313,314,333,334,343,344,373,374,403,404,433,434]]

GridLocation = Tuple[int, int]

class SquareGrid:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.walls: list[GridLocation] = []
    
    def in_bounds(self, id: GridLocation) -> bool:
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height
    
    def passable(self, id: GridLocation) -> bool:
        return id not in self.walls
    
    def neighbors(self, id: GridLocation) -> Iterator[GridLocation]:
        (x, y) = id
        neighbors = [(x+1, y), (x-1, y), (x, y-1), (x, y+1)] # E W N S
        if (x + y) % 2 == 0: neighbors.reverse() # S N W E
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return results

class WeightedGraph(Graph):
    def cost(self, from_id: Location, to_id: Location) -> float: pass

class GridWithWeights(SquareGrid):
    def __init__(self, width: int, height: int):
        super().__init__(width, height)
        self.weights: dict[GridLocation, float] = {}
    
    def cost(self, from_node: GridLocation, to_node: GridLocation) -> float:
        return self.weights.get(to_node, 1)

diagram4 = GridWithWeights(10, 10)
diagram4.walls = [(1, 7), (1, 8), (2, 7), (2, 8), (3, 7), (3, 8)]
diagram4.weights = {loc: 5 for loc in [(3, 4), (3, 5), (4, 1), (4, 2),
                                       (4, 3), (4, 4), (4, 5), (4, 6),
                                       (4, 7), (4, 8), (5, 1), (5, 2),
                                       (5, 3), (5, 4), (5, 5), (5, 6),
                                       (5, 7), (5, 8), (6, 2), (6, 3),
                                       (6, 4), (6, 5), (6, 6), (6, 7),
                                       (7, 3), (7, 4), (7, 5)]}

import heapq

class PriorityQueue:
    def __init__(self):
        self.elements: list[tuple[float, T]] = []
    
    def empty(self) -> bool:
        return not self.elements
    
    def put(self, item: T, priority: float):
        heapq.heappush(self.elements, (priority, item))
    
    def get(self) -> T:
        return heapq.heappop(self.elements)[1]

def dijkstra_search(graph: WeightedGraph, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: dict[Location, Optional[Location]] = {}
    cost_so_far: dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current: Location = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def reconstruct_path(came_from: dict[Location, Location],
                     start: Location, goal: Location) -> list[Location]:

    current: Location = goal
    path: list[Location] = []
    if goal not in came_from: # no path was found
        return []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start) # optional
    path.reverse() # optional
    return path

diagram_nopath = GridWithWeights(10, 10)
diagram_nopath.walls = [(5, row) for row in range(10)]

def heuristic(a: GridLocation, b: GridLocation) -> float:
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

def a_star_search(graph: WeightedGraph, start: Location, goal: Location):
    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from: dict[Location, Optional[Location]] = {}
    cost_so_far: dict[Location, float] = {}
    came_from[start] = None
    cost_so_far[start] = 0
    
    while not frontier.empty():
        current: Location = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                frontier.put(next, priority)
                came_from[next] = current
    
    return came_from, cost_so_far

def breadth_first_search(graph: Graph, start: Location, goal: Location):
    frontier = Queue()
    frontier.put(start)
    came_from: dict[Location, Optional[Location]] = {}
    came_from[start] = None
    
    while not frontier.empty():
        current: Location = frontier.get()
        
        if current == goal:
            break
        
        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
    
    return came_from

class SquareGridNeighborOrder(SquareGrid):
    def neighbors(self, id):
        (x, y) = id
        neighbors = [(x + dx, y + dy) for (dx, dy) in self.NEIGHBOR_ORDER]
        results = filter(self.in_bounds, neighbors)
        results = filter(self.passable, results)
        return list(results)

def test_with_custom_order(neighbor_order):
    if neighbor_order:
        g = SquareGridNeighborOrder(30, 15)
        g.NEIGHBOR_ORDER = neighbor_order
    else:
        g = SquareGrid(30, 15)
    g.walls = DIAGRAM1_WALLS
    start, goal = (8, 7), (27, 2)
    came_from = breadth_first_search(g, start, goal)
    draw_grid(g, path=reconstruct_path(came_from, start=start, goal=goal),
              point_to=came_from, start=start, goal=goal)

class GridWithAdjustedWeights(GridWithWeights):
    def cost(self, from_node, to_node):
        prev_cost = super().cost(from_node, to_node)
        nudge = 0
        (x1, y1) = from_node
        (x2, y2) = to_node
        if (x1 + y1) % 2 == 0 and x2 != x1: nudge = 1
        if (x1 + y1) % 2 == 1 and y2 != y1: nudge = 1
        return prev_cost + 0.001 * nudge

start, goal = (1, 4), (8, 3)
came_from, cost_so_far = a_star_search(diagram4, start, goal)

print("Here is the puzzle:")

draw_grid(diagram4, point_to=came_from, start=start, goal=goal)
print()

print("Here is the optimal path using Manhattan distance:")

draw_grid(diagram4, path=reconstruct_path(came_from, start=start, goal=goal))

print("Here are the distances from point A to Z:")

start, goal = (1, 4), (8, 3)
came_from, cost_so_far = a_star_search(diagram4, start, goal)
draw_grid(diagram4, number=cost_so_far, start=start, goal=goal)






Genetic algorithms T1:







# Python3 program to create target string, starting from 
# random string using Genetic Algorithm 

import random 

# Number of individuals in each generation 
POPULATION_SIZE = 100

# Valid genes 
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP 
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated 
TARGET = "Hello there."

class Individual(object): 
	''' 
	Class representing individual in population 
	'''
	def __init__(self, chromosome): 
		self.chromosome = chromosome 
		self.fitness = self.cal_fitness() 

	@classmethod
	def mutated_genes(self): 
		''' 
		create random genes for mutation 
		'''
		global GENES 
		gene = random.choice(GENES) 
		return gene 

	@classmethod
	def create_gnome(self): 
		''' 
		create chromosome or string of genes 
		'''
		global TARGET 
		gnome_len = len(TARGET) 
		return [self.mutated_genes() for _ in range(gnome_len)] 

	def mate(self, par2): 
		''' 
		Perform mating and produce new offspring 
		'''

		# chromosome for offspring 
		child_chromosome = [] 
		for gp1, gp2 in zip(self.chromosome, par2.chromosome):	 

			# random probability 
			prob = random.random() 

			# if prob is less than 0.45, insert gene 
			# from parent 1 
			if prob < 0.45: 
				child_chromosome.append(gp1) 

			# if prob is between 0.45 and 0.90, insert 
			# gene from parent 2 
			elif prob < 0.90: 
				child_chromosome.append(gp2) 

			# otherwise insert random gene(mutate), 
			# for maintaining diversity 
			else: 
				child_chromosome.append(self.mutated_genes()) 

		# create new Individual(offspring) using 
		# generated chromosome for offspring 
		return Individual(child_chromosome) 

	def cal_fitness(self): 
		''' 
		Calculate fitness score, it is the number of 
		characters in string which differ from target 
		string. 
		'''
		global TARGET 
		fitness = 0
		for gs, gt in zip(self.chromosome, TARGET): 
			if gs != gt: fitness+= 1
		return fitness 

# Driver code 
def main(): 
	global POPULATION_SIZE 

	#current generation 
	generation = 1

	found = False
	population = [] 

	# create initial population 
	for _ in range(POPULATION_SIZE): 
				gnome = Individual.create_gnome() 
				population.append(Individual(gnome)) 

	while not found: 

		# sort the population in increasing order of fitness score 
		population = sorted(population, key = lambda x:x.fitness) 

		# if the individual having lowest fitness score ie. 
		# 0 then we know that we have reached to the target 
		# and break the loop 
		if population[0].fitness <= 0: 
			found = True
			break

		# Otherwise generate new offsprings for new generation 
		new_generation = [] 

		# Perform Elitism, that mean 10% of fittest population 
		# goes to the next generation 
		s = int((10*POPULATION_SIZE)/100) 
		new_generation.extend(population[:s]) 

		# From 50% of fittest population, Individuals 
		# will mate to produce offspring 
		s = int((90*POPULATION_SIZE)/100) 
		for _ in range(s): 
			parent1 = random.choice(population[:50]) 
			parent2 = random.choice(population[:50]) 
			child = parent1.mate(parent2) 
			new_generation.append(child) 

		population = new_generation 

		print("Generation: {}\tString: {}\tFitness: {}".\ 
			format(generation, 
			"".join(population[0].chromosome), 
			population[0].fitness)) 

		generation += 1

	
	print("Generation: {}\tString: {}\tFitness: {}".\ 
		format(generation, 
		"".join(population[0].chromosome), 
		population[0].fitness)) 

if __name__ == '__main__': 
	main()









Genetic algorithms T2:






# Python3 program to create target string, starting from 
# random string using Genetic Algorithm 

import random 

# Number of individuals in each generation 
POPULATION_SIZE = 150

# Valid genes 
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP 
QRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''

# Target string to be generated 
TARGET = "Today was a long day."

class Individual(object): 
  ''' 
  Class representing individual in population 
  '''
  def __init__(self, chromosome): 
    self.chromosome = chromosome 
    self.fitness = self.cal_fitness() 

  @classmethod
  def mutated_genes(self): 
    ''' 
    create random genes for mutation 
    '''
    global GENES 
    gene = random.choice(GENES) 
    return gene 

  @classmethod
  def create_gnome(self): 
    ''' 
    create chromosome or string of genes 
    '''
    global TARGET 
    gnome_len = len(TARGET) 
    return [self.mutated_genes() for _ in range(gnome_len)] 

  def mate(self, par2): 
    ''' 
    Perform mating and produce new offspring 
    '''

    # chromosome for offspring 
    child_chromosome = [] 
    for gp1, gp2 in zip(self.chromosome, par2.chromosome):   

      # random probability 
      prob = 0.75 

      # if prob is less than 0.45, insert gene 
      # from parent 1 
      if prob < 0.45: 
        child_chromosome.append(gp1) 

      # if prob is between 0.45 and 0.90, insert 
      # gene from parent 2 
      elif prob < 0.90: 
        child_chromosome.append(gp2) 

      # otherwise insert random gene(mutate), 
      # for maintaining diversity 
      #else: 
        child_chromosome.append(self.mutated_genes()) 

    # create new Individual(offspring) using 
    # generated chromosome for offspring

    return Individual(child_chromosome)

  def cal_fitness(self): 
    ''' 
    Calculate fitness score, it is the number of 
    characters in string which differ from target 
    string. 
    '''
    global TARGET 
    fitness = 0
    for gs, gt in zip(self.chromosome, TARGET): 
      if gs != gt: fitness+= 1
    return fitness 

# Driver code 
def main(): 
  global POPULATION_SIZE 

  #current generation 
  generation = 1

  found = False
  population = [] 

  # create initial population 
  for _ in range(POPULATION_SIZE): 
        gnome = Individual.create_gnome() 
        population.append(Individual(gnome)) 

  while not found: 

    # sort the population in increasing order of fitness score 
    population = sorted(population, key = lambda x:x.fitness) 

    # if the individual having lowest fitness score ie. 
    # 0 then we know that we have reached to the target 
    # and break the loop 
    if population[0].fitness <= 0: 
      found = True
      break

    # Otherwise generate new offsprings for new generation 
    new_generation = [] 

    # Perform Elitism, that mean 10% of fittest population 
    # goes to the next generation 
    s = int((10*POPULATION_SIZE)/100) 
    new_generation.extend(population[:s]) 

    # From 50% of fittest population, Individuals 
    # will mate to produce offspring 
    s = int((90*POPULATION_SIZE)/100) 
    for _ in range(s): 
      parent1 = random.choice(population[:50]) 
      parent2 = random.choice(population[:50]) 
      child = parent1.mate(parent2) 
      new_generation.append(child) 

    population = new_generation 

    print("Generation: {}\tString: {}\tFitness: {}".\
      format(generation, 
      "".join(population[0].chromosome), 
      population[0].fitness)) 

    generation += 1

  
  print("Generation: {}\tString: {}\tFitness: {}".\
    format(generation, 
    "".join(population[0].chromosome), 
    population[0].fitness)) 

if __name__ == '__main__': 
  main() 











K-mean clustering:








import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from numpy.random import uniform
from sklearn.datasets import make_blobs
import seaborn as sns
import random
def euclidean(point, data):
    """
    Euclidean distance between point & data.
    Point has dimensions (m,), data has dimensions (n,m), and output will be of size (n,).
    """
    return np.sqrt(np.sum((point - data)**2, axis=1))
class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
    def fit(self, X_train):
        # Initialize the centroids, using the "k-means++" method, where a random datapoint is selected as the first,
        # then the rest are initialized w/ probabilities proportional to their distances to the first
        # Pick a random point from train data for first centroid
        self.centroids = [random.choice(X_train)]
        for _ in range(self.n_clusters-1):
            # Calculate distances from points to the centroids
            dists = np.sum([euclidean(centroid, X_train) for centroid in self.centroids], axis=0)
            # Normalize the distances
            dists /= np.sum(dists)
            # Choose remaining points based on their distances
            new_centroid_idx, = np.random.choice(range(len(X_train)), size=1, p=dists)
            self.centroids += [X_train[new_centroid_idx]]
        # This initial method of randomly selecting centroid starts is less effective
        # min_, max_ = np.min(X_train, axis=0), np.max(X_train, axis=0)
        # self.centroids = [uniform(min_, max_) for _ in range(self.n_clusters)]
        # Iterate, adjusting centroids until converged or until passed max_iter
        iteration = 0
        prev_centroids = None
        while np.not_equal(self.centroids, prev_centroids).any() and iteration < self.max_iter:
            # Sort each datapoint, assigning to nearest centroid
            sorted_points = [[] for _ in range(self.n_clusters)]
            for x in X_train:
                dists = euclidean(x, self.centroids)
                centroid_idx = np.argmin(dists)
                sorted_points[centroid_idx].append(x)
            # Push current centroids to previous, reassign centroids as mean of the points belonging to them
            prev_centroids = self.centroids
            self.centroids = [np.mean(cluster, axis=0) for cluster in sorted_points]
            for i, centroid in enumerate(self.centroids):
                if np.isnan(centroid).any():  # Catch any np.nans, resulting from a centroid having no points
                    self.centroids[i] = prev_centroids[i]
            iteration += 1
    def evaluate(self, X):
        centroids = []
        centroid_idxs = []
        for x in X:
            dists = euclidean(x, self.centroids)
            centroid_idx = np.argmin(dists)
            centroids.append(self.centroids[centroid_idx])
            centroid_idxs.append(centroid_idx)
        return centroids, centroid_idxs
# Create a dataset of 2D distributions
centers = 5
X_train, true_labels = make_blobs(n_samples=100, centers=centers, random_state=42)
X_train = StandardScaler().fit_transform(X_train)
# Fit centroids to dataset
kmeans = KMeans(n_clusters=centers)
kmeans.fit(X_train)
# View results
class_centers, classification = kmeans.evaluate(X_train)
sns.scatterplot(x=[X[0] for X in X_train],
                y=[X[1] for X in X_train],
                hue=true_labels,
                style=classification,
                palette="deep",
                legend=None
                )
plt.plot([x for x, _ in kmeans.centroids],
         [y for _, y in kmeans.centroids],
         'k+',
         markersize=10,
         )
plt.show()










KNN:







#Module Importing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
# Dataset creation, centers = classes, n_features = characteristics (Changed dataset)
X, y = make_blobs(n_samples = 790, n_features = 5, centers = 2, cluster_std = 3.5, random_state = 10)
#Dataset visualization
plt.figure(figsize = (10,10))
plt.scatter(X[:,0], X[:,1], c=y, s=100,edgecolors = 'black')
plt.show()
#Splitting data into training and testing set, 75% for the train set and 25% for the test set (Changed random state from 0 to 4)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 4)
# KNN implementation, Two variants (Changed neighbors)
knn3 = KNeighborsClassifier(n_neighbors = 3)
knn7 = KNeighborsClassifier(n_neighbors=7)
# Predictions for KNN Classifiers (n=3,7)
knn3.fit(X_train, y_train)
knn7.fit(X_train, y_train)
# Accuracy prediction
y_pred_3 = knn3.predict(X_test)
y_pred_7 = knn7.predict(X_test)
from sklearn.metrics import accuracy_score
print("Accuracy with k=3", accuracy_score(y_test, y_pred_3)*100)
print("Accuracy with k=7", accuracy_score(y_test, y_pred_7)*100)
# Predictions visualization
plt.figure(figsize = (15,5))
plt.subplot(1,2,1)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_3, marker= '*', s=100,edgecolors='black')
plt.title("Predicted values with k=3", fontsize=20)

plt.subplot(1,2,2)
plt.scatter(X_test[:,0], X_test[:,1], c=y_pred_7, marker= '*', s=100,edgecolor= 'black')
plt.title("Predicted values with k=7", fontsize=20)
plt.show()







Linear regression:






#Import libraries
import matplotlib.pyplot as plt
from scipy import stats
#Import data
profit = [1, 2, 3, 4, 5, 6]
population = [50, 100, 150, 40, 170, 90]
#Visualize the data
plt.figure()
plt.scatter(population, profit)
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Data Visualization")
slope, intercept, r, p, std_err = stats.linregress(population, profit)
def linearRegression(x):
  return slope * x + intercept
LR = list(map(linearRegression, population))
print("LR Model Values:",LR)
plt.figure()
plt.scatter(population, profit)
plt.plot(population, LR)
plt.xlabel("Population")
plt.ylabel("Profit")
plt.title("Model Fit Line")
plt.show()
Newpop = 300
pf = linearRegression(Newpop)
print("The prediction of profit for 300 people is: ", pf)







Naive Bayes Classifier:








from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
#Generating the Dataset
X, y = make_classification(
    n_features=10,
    n_classes=5,
    n_samples=1000,
    n_informative=5,
    random_state=4,
    n_clusters_per_class=3,
)
plt.scatter(X[:, 0], X[:, 1], c=y, marker="*");
#Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.66, random_state=300
)
#Model Building and Training
from sklearn.naive_bayes import GaussianNB
# Build a Gaussian Classifier
model = GaussianNB()
# Model training
model.fit(X_train, y_train)

# Predict Output
predicted = model.predict([X_test[6]])

print("Actual Value:", y_test[6])
print("Predicted Value:", predicted[0])
#Model Evaluation
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)

y_pred = model.predict(X_test)
accuray = accuracy_score(y_pred, y_test)
f1 = f1_score(y_pred, y_test, average="weighted")
print("Accuracy:", accuray)
print("F1 Score:", f1)
labels = [0,1,2]
#visualize the Confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot();







Final project (text data):







import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset with specified data types to handle mixed types
text_data = pd.read_csv('/content/sample_data/Amazon_Unlocked_Mobile2.csv', dtype={'Product Name': str, 'Brand Name': str, 'Price': str, 'Rating': str, 'Reviews': str, 'Review Votes': str})

# Handle missing values
text_data['Reviews'] = text_data['Reviews'].fillna('')
text_data['Rating'] = text_data['Rating'].fillna('0')  # Fill missing ratings with '0' or another placeholder value

# Convert 'Rating' to numeric, forcing errors to NaN and then filling them
text_data['Rating'] = pd.to_numeric(text_data['Rating'], errors='coerce').fillna(0).astype(int)

# Assume 'Reviews' is the text and 'Rating' is the target
X = text_data['Reviews']
y = text_data['Rating']

# Preprocess text data
vectorizer = TfidfVectorizer(stop_words='english')
X_transformed = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_transformed, y, test_size=0.2, random_state=42)

# Function to evaluate models
def evaluate_model(y_test, y_pred, model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"{model_name} Confusion Matrix:\n", conf_matrix)
    print(f"{model_name} Accuracy: {acc}")
    print(f"{model_name} Precision: {prec}")
    print(f"{model_name} Recall: {rec}")
    print(f"{model_name} F1-Score: {f1}")
    return conf_matrix, acc, prec, rec, f1

# Initialize models
models = {
    "Naive Bayes": MultinomialNB(),
    "MLP Classifier": MLPClassifier(hidden_layer_sizes=(100,), max_iter=300),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100)
}

results = {}

# Train and evaluate models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[model_name] = evaluate_model(y_test, y_pred, model_name)

# Display results
results_df = pd.DataFrame(results, index=['Confusion Matrix', 'Accuracy', 'Precision', 'Recall', 'F1-Score'])
print(results_df.T)








Final project (numeric data):







import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load dataset with specified encoding to handle BOM
numeric_data = pd.read_csv('/content/drive/MyDrive/Datasets/Online_Retail.csv', encoding='latin1')

# Strip column names to remove any leading/trailing spaces
numeric_data.columns = numeric_data.columns.str.strip()

# Create a 'TotalPrice' column
numeric_data['TotalPrice'] = numeric_data['Quantity'] * numeric_data['UnitPrice']

# Define a threshold for high value transactions
threshold = 1000  # Adjust the threshold based on your criteria
numeric_data['HighValueTransaction'] = numeric_data['TotalPrice'] > threshold

# Define the target column
target_column = 'HighValueTransaction'

# Preprocess dataset
# Separate numerical and categorical columns
numeric_cols = numeric_data.select_dtypes(include=np.number).columns.tolist()
categorical_cols = numeric_data.select_dtypes(exclude=np.number).columns.tolist()

# Fill missing values
numeric_data[numeric_cols] = numeric_data[numeric_cols].fillna(numeric_data[numeric_cols].mean())
numeric_data[categorical_cols] = numeric_data[categorical_cols].fillna('missing')

# Encode categorical variables
for col in categorical_cols:
    le = LabelEncoder()
    numeric_data[col] = le.fit_transform(numeric_data[col])

# Features and target variable
X = numeric_data.drop(target_column, axis=1)  # Features
y = numeric_data[target_column]  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Function to evaluate and print model performance
def evaluate_model(y_test, y_pred, model_name):
    conf_matrix = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='weighted')
    rec = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"{model_name} Confusion Matrix:\n", conf_matrix)
    print(f"{model_name} Accuracy:", acc)
    print(f"{model_name} Precision:", prec)
    print(f"{model_name} Recall:", rec)
    print(f"{model_name} F1-Score:", f1)

    return acc, prec, rec, f1

# KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
acc_knn, prec_knn, rec_knn, f1_knn = evaluate_model(y_test, y_pred_knn, "KNN")

# SVM
svm = SVC()
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
acc_svm, prec_svm, rec_svm, f1_svm = evaluate_model(y_test, y_pred_svm, "SVM")

# MLP
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=300, alpha=0.0001, solver='adam', random_state=42)
mlp.fit(X_train, y_train)
y_pred_mlp = mlp.predict(X_test)
acc_mlp, prec_mlp, rec_mlp, f1_mlp = evaluate_model(y_test, y_pred_mlp, "MLP")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf, prec_rf, rec_rf, f1_rf = evaluate_model(y_test, y_pred_rf, "Random Forest")

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
acc_gb, prec_gb, rec_gb, f1_gb = evaluate_model(y_test, y_pred_gb, "Gradient Boosting")

# Improved Ensemble with Hyper-parameter Tuning
param_grid = {
    'n_estimators': [100, 200],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [4, 5, 6],
    'criterion': ['gini', 'entropy']
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
best_rf = grid_search.best_estimator_
y_pred_best_rf = best_rf.predict(X_test)
acc_best_rf, prec_best_rf, rec_best_rf, f1_best_rf = evaluate_model(y_test, y_pred_best_rf, "Improved Random Forest")

# Compile results into a DataFrame
results = pd.DataFrame({
    'Model': ['KNN', 'SVM', 'MLP', 'Random Forest', 'Gradient Boosting', 'Improved RF'],
    'Accuracy': [acc_knn, acc_svm, acc_mlp, acc_rf, acc_gb, acc_best_rf],
    'Precision': [prec_knn, prec_svm, prec_mlp, prec_rf, prec_gb, prec_best_rf],
    'Recall': [rec_knn, rec_svm, rec_mlp, rec_rf, rec_gb, rec_best_rf],
    'F1-Score': [f1_knn, f1_svm, f1_mlp, f1_rf, f1_gb, f1_best_rf]
})

print("\nResults Summary:\n", results)








LAB 2:
Question:
Write a Python program to count the elements in a list until an element is a tuple. Hint: Use isinstance(object, type).
def count_until_tuple(lst):
    count = 0
    for element in lst:
        if isinstance(element, tuple):
            break
        count += 1
    return count

# Example usage:
my_list = [1, 2, 3, (4, 5), 6, 7]
print("Number of elements until tuple:", count_until_tuple(my_list))
Question:
Count the number of strings where the string length is 2 or more and the first and last character are same from a given list of strings.Sample List: ['abc', ' xyz', 'aba', '1221', 'xyzzyx', 'aa', '122'] Expected Result: 4
def count_strings_with_same_first_and_last(strings):
    count = 0
    for string in strings:
        if len(string) >= 2 and string[0] == string[-1]:
            count += 1
    return count

# Sample list of strings
sample_list = ['abc', ' xyz', 'aba', '1221', 'xyzzyx', 'aa', '122']

# Count strings meeting the criteria
result = count_strings_with_same_first_and_last(sample_list)
print("Number of strings where first and last characters are the same:", result)


Question:
Write a Python program to remove an empty tuple(s) from a list of tuples. Sample L = [(), (), (",), ('a', 'b'),(), (a', 'b', 'c'), () , ('d')] Result = [(",), (a', 'b'), (a', 'b', 'c'), ('d')]
def remove_empty_tuples(lst):
    return [tup for tup in lst if tup]

# Sample list of tuples
L = [(), (), (",), ('a', 'b'),(), ('a', 'b', 'c'), () , ('d')]

# Remove empty tuples
result = remove_empty_tuples(L)
print("Result:", result)

Question:
Write a Python script to generate and print a dictionary that contains a number (between 1 and n) in the form ('x', x*x). Sample: Input a number 4 Output: {'1': 1, '2': 4, '3': 9, '4': 16}
def generate_square_dictionary(n):
    square_dict = {}
    for x in range(1, n + 1):
        square_dict[str(x)] = x * x
    return square_dict

# Input number from the user
n = int(input("Input a number: "))

# Generate and print the dictionary
result = generate_square_dictionary(n)
print("Output:", result)


Question:
Create a dictionary of phone numbers. Find the name 'usman' and get his phone number and country code.
Sample:
Enter Name: Touqeer Ali
- Information.
Name: Touqeer Ali
Country Code: +92
Phone No: 3428650577

def find_contact(contacts, name):
    for contact in contacts:
        if contact['Name'].lower() == name.lower():
            return contact

# Dictionary of phone numbers
phone_numbers = [
    {'Name': 'Touqeer Ali', 'Country Code': '+92', 'Phone No': '3428650577'},
    {'Name': 'Usman', 'Country Code': '+1', 'Phone No': '1234567890'},
    {'Name': 'John Doe', 'Country Code': '+44', 'Phone No': '9876543210'}
]

# Input name from the user
name = input("Enter Name: ")

# Find the contact information
contact_info = find_contact(phone_numbers, name)

# Print the information if found
if contact_info:
    print("- Information.")
    print("Name:", contact_info['Name'])
    print("Country Code:", contact_info['Country Code'])
    print("Phone No:", contact_info['Phone No'])
else:
    print("Name not found in contacts.")

LAB 3:
A loop may be used to access all elements of an array.
# define a list with three elements
myList = ["abc", 123, "xyz"]
# use the len( ) function to determine the number of elements in the list 
# display this value to the screen
myListLength = len(myList)
print("\nThe length of myList is", myListLength)
# display the values of the list to the screen 
print("\nmyList contains the following values:") 
index = 0
while (index < myListLength):
	print("index: ", index, "value: ", myList[index]) 
	index = index + 1

While defining a list in the code is all well and good, it would be even better to be able to generate the list 
with data from the user or a file. The append( )method provides this capability. Add the following to 
the end of the program code in order to add to myList.
# prompt the user for two more elements for the list 
myList = ["abc", 123, "xyz"]
value1 = float(input("\nEnter a decimal value: ")) 
value2 = input("Enter your name: ")

# add the values to the list 
myList.append(value1) 
myList.append(value2)

# display the values of the list to the screen 
print("\nmyList contains the following values:") 
index = 0
myListLength = len(myList)
while (index < myListLength):
    print("index:", index, "value:", myList[index]) 
    index = index + 1

write a python program to implement list operations(add,append,extend and delete)
def add_to_list(lst, item):
    lst.append(item)
    return lst
def append_to_list(lst, item):
    lst.append(item)
    return lst
def extend_list(lst, items):
    lst.extend(items)
    return lst
def delete_from_list(lst, item):
    lst.remove(item)
    return lst
my_list = [1, 2, 3]
my_list = add_to_list(my_list, 4)
print(my_list)
my_list = append_to_list(my_list, 5)
print(my_list)
my_list = extend_list(my_list, [6, 7, 8])
print(my_list)
my_list = delete_from_list(my_list, 5)
print(my_list)


Use for loops to make a turtle draw these regular polygons (regular means all sides the same lengths, all angles the same):
a. An equilateral triangle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

# Function to draw an equilateral triangle
def draw_equilateral_triangle():
    fig, ax = plt.subplots()
    triangle = Polygon(np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]]), closed=True)
    ax.add_patch(triangle)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, np.sqrt(3)/2 + 0.1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.show()

# Draw an equilateral triangle
draw_equilateral_triangle()
b. square 
import matplotlib.pyplot as plt

# Function to draw a square
def draw_square():
    fig, ax = plt.subplots()
    square = plt.Rectangle((0, 0), 1, 1, fill=None, edgecolor='black')
    ax.add_patch(square)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')
    plt.show()

# Draw a square
draw_square()

import matplotlib.pyplot as plt
import numpy as np

# Function to draw a regular polygon
def draw_polygon(sides):
    angle = 360 / sides
    points = []
    for i in range(sides):
        x = np.cos(np.deg2rad(i * angle))
        y = np.sin(np.deg2rad(i * angle))
        points.append((x, y))
    points.append(points[0])  # Connect the last point to the first point to close the shape
    return list(zip(*points))  # Separate x and y coordinates

# Draw a hexagon
hexagon_x, hexagon_y = draw_polygon(6)
plt.plot(hexagon_x, hexagon_y, 'b-')
plt.title('Hexagon')
plt.axis('equal')
plt.show()

# Draw an octagon
octagon_x, octagon_y = draw_polygon(8)
plt.plot(octagon_x, octagon_y, 'r-')
plt.title('Octagon')
plt.axis('equal')
plt.show()

Write a python program to print the multiplication table for the given number?
def print_multiplication_table(number):
    print(f"Multiplication Table for {number}:")
    for i in range(1, 11):
        print(f"{number} x {i} = {number * i}")

# Get input from the user
number = int(input("Enter a number: "))

# Call the function to print the multiplication table
print_multiplication_table(number)

Given a list iterate it and display numbers which are divisible by 5 and if you find number greater than 150 stop the loop 
iteration
list1 = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
list1 = [12, 15, 32, 42, 55, 75, 122, 132, 150, 180, 200]
print("Numbers divisible by 5 and less than or equal to 150:")
for num in list1:
    if num > 150:
        break
    if num % 5 == 0:
        print(num)
LAB 4:
BFS Search 
from collections import defaultdict, deque

class Graph:
    def __init__(self):
        self.graph = defaultdict(list)
        
    def add_edge(self, u, v):
        self.graph[u].append(v)
        
    def bfs(self, start):
        visited = set()
        queue = deque([start])
        
        while queue:
            vertex = queue.popleft()
            if vertex not in visited:
                print(vertex, end=' ')
                visited.add(vertex)
                for neighbor in self.graph[vertex]:
                    if neighbor not in visited:
                        queue.append(neighbor)

# Example usage:
if __name__ == "__main__":
    g = Graph()
    g.add_edge(0, 1)
    g.add_edge(0, 2)
    g.add_edge(1, 2)
    g.add_edge(2, 0)
    g.add_edge(2, 3)
    g.add_edge(3, 3)
    
    print("BFS starting from vertex 2:")
    g.bfs(2)

DFS:
 Depth First Search (DFS) traversal algorithm for a graph represented using an adjacency list
from collections import defaultdict

class Graph:
    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited):
        # Mark the current node as visited
        # and print it
        visited.add(v)
        print(v, end=' ')
        # Recur for all the vertices
        # adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses
    # recursive DFSUtil()
    def DFS(self, v):
        # Create a set to store visited vertices
        visited = set()
        # Call the recursive helper function
        # to print DFS traversal
        self.DFSUtil(v, visited)

# Driver code
# Create a graph given
# in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)
print("Following is DFS from (starting from vertex 2)")
g.DFS(2)

 Depth First Search (DFS) algorithm for a directed graph represented using an adjacency list.
from collections import defaultdict

# This class represents a
# directed graph using adjacency
# list representation
class Graph:
    # Constructor
    def __init__(self):
        # default dictionary to store graph
        self.graph = defaultdict(list)

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A function used by DFS
    def DFSUtil(self, v, visited):
        # Mark the current node as visited and print it
        visited.add(v)
        print(v, end=" ")

        # Recur for all the vertices adjacent to this vertex
        for neighbour in self.graph[v]:
            if neighbour not in visited:
                self.DFSUtil(neighbour, visited)

    # The function to do DFS traversal. It uses recursive DFSUtil()
    def DFS(self):
        # Create a set to store all visited vertices
        visited = set()

        # Call the recursive helper function to print DFS traversal starting from all vertices one by one
        for vertex in list(self.graph):
            if vertex not in visited:
                self.DFSUtil(vertex, visited)

# Driver code
# Create a graph given in the above diagram
g = Graph()
g.addEdge(0, 1)
g.addEdge(0, 2)
g.addEdge(1, 2)
g.addEdge(2, 0)
g.addEdge(2, 3)
g.addEdge(3, 3)

print("Following is Depth First Traversal:")
g.DFS()
LAB 5:
Write a python program for Depth First Search for INORDER,
PREORDER and POSTORDER Traversing Methods.
class TreeNode:
    def __init__(self, key):
        self.left = None
        self.right = None
        self.val = key

def inorder(root):
    if root:
        inorder(root.left)
        print(root.val, end=" ")
        inorder(root.right)

def preorder(root):
    if root:
        print(root.val, end=" ")
        preorder(root.left)
        preorder(root.right)

def postorder(root):
    if root:
        postorder(root.left)
        postorder(root.right)
        print(root.val, end=" ")

# Driver code
if __name__ == "__main__":
    root = TreeNode(1)
    root.left = TreeNode(2)
    root.right = TreeNode(3)
    root.left.left = TreeNode(4)
    root.left.right = TreeNode(5)

    print("Inorder traversal:")
    inorder(root)

    print("\nPreorder traversal:")
    preorder(root)

    print("\nPostorder traversal:")
    postorder(root)

LAB 6:
A* search:
from collections import deque

class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    def h(self, n):
        H = {'A': 1, 'B': 1, 'C': 1, 'D': 1}
        return H[n]

    def a_star_algorithm(self, start_node, stop_node):
        open_list = set([start_node])
        closed_list = set([])
        g = {start_node: 0}
        parents = {start_node: start_node}

        while open_list:
            n = min(open_list, key=lambda x: g[x] + self.h(x))

            if n == stop_node:
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                reconst_path.append(start_node)
                reconst_path.reverse()
                print('Path found: {}'.format(reconst_path))
                return reconst_path

            for (m, weight) in self.get_neighbors(n):
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                else:
                    if g[m] > g[n] + weight:
                        g[m] = g[n] + weight
                        parents[m] = n
                        if m in closed_list:
                            closed_list.remove(m)
                            open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
        print('Path does not exist!')
        return None

# Create an instance of the Graph class with the provided adjacency list
adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}

g = Graph(adjacency_list)

# Test the A* algorithm
start_node = 'A'
stop_node = 'D'
print(f"Finding path from {start_node} to {stop_node}:")
g.a_star_algorithm(start_node, stop_node)


Write Heuristic Greedy Best search python program.
from queue import PriorityQueue
class Graph:
    def __init__(self, adjacency_list):
        self.adjacency_list = adjacency_list

    def get_neighbors(self, v):
        return self.adjacency_list[v]

    # Heuristic function
    def h(self, n):
        H = {'A': 10, 'B': 5, 'C': 7, 'D': 3}
        return H[n]

    def greedy_best_first_search(self, start_node, stop_node):
        visited = set()
        pq = PriorityQueue()
        pq.put((0, start_node))

        while not pq.empty():
            cost, node = pq.get()
            visited.add(node)
            print(node, end=" ")

            if node == stop_node:
                print("\nGoal reached!")
                return True

            for (neighbour, _) in self.get_neighbors(node):
                if neighbour not in visited:
                    priority = self.h(neighbour)
                    pq.put((priority, neighbour))

        print("\nGoal not reachable!")
        return False

# Define the adjacency list for the graph
adjacency_list = {
    'A': [('B', 1), ('C', 3), ('D', 7)],
    'B': [('D', 5)],
    'C': [('D', 12)]
}

# Create an instance of the Graph class
g = Graph(adjacency_list)

# Test the Greedy Best-First Search algorithm
start_node = 'A'
stop_node = 'D'
print(f"Greedy Best-First Search from {start_node} to {stop_node}:")
g.greedy_best_first_search(start_node, stop_node)





Create Root
class Node:
def __init__(self, data): 
 self.left = None 
 self.right = None 
 self.data = data
def PrintTree(self): 
print(self.data , end = " ") 
root = Node(10)
LAB # 05
print("Tree is: " , end = " ") 
root.PrintTree()
Inserting into a Tree
# Compare the new value with the parent node
if self.data:
if data < self.data:
if self.left is None:
self.left = Node(data)
else:
self.left.insert(data)
elif data > self.data:
if self.right is None:
self.right = Node(data)
else:
self.right.insert(data)
else:
self.data = data








# Define the tree structure
tree = {
    1: [2, 5, 7],
    2: [3, 4],
    5: [6],
    7: [8, 9]
}

# Perform Depth-First Search (DFS) recursively
def dfs(node, visited):
    if node not in visited:
        print(node)
        visited.add(node)
        if node in tree:
            for neighbor in tree[node]:
                dfs(neighbor, visited)

# Start DFS from the root node (1)
print("Depth-First Search (DFS):")
dfs(1, set())
