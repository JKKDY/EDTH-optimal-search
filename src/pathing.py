import numpy as np
def snake_fill(polygon:np.ndarray, num_waypoints:int):
  pass

def zigzag_fill(polygon:np.ndarray, num_waypoints:int):
  """
  Expect a (4,2) numpy array of corner point that this drone has to cover in clockwise/counter clockwise order
  """
  waypoints = np.zeros((num_waypoints,2))
  waypoints[0] = polygon[0]
  
  for i in range(num_waypoints):
    percent = i / num_waypoints
    edge1 = (1.0-percent)*polygon[0] +  percent*polygon[1]
    edge2 = (1.0-percent)*polygon[3] +  percent*polygon[2]
    
    waypoints[i] = edge1 if i % 2 == 0 else edge2
  return waypoints

def spiral_fill(polygon:np.ndarray, num_waypoints:int):
  return 


if __name__ == "__main__":
  polygon = np.array([[0,0],[0,10],[10,10],[10,0]])
  num_waypoints = 10
  waypoints = snake_fill(polygon, num_waypoints)
  print(waypoints)

  waypoints = zigzag_fill(polygon, num_waypoints)
  print(waypoints)

  waypoints = spiral_fill(polygon, num_waypoints)
  print(waypoints)