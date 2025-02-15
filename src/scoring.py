import numpy as np

def score(p_discovery, p_prior=None):
  """
  p_discovery: probability of discovering a target at this point, if it exists 
  p_prior: expected probability that a target is there

  Both quantities can be scalar (N,M) or vector valued (N,M, DIRS) where N,M are 
  the size of the map and DIRS are the number of directions the discovery metric 
  is quantized into.  
  """
  def activation_function(x):
    return 1.0 - np.exp(-x)
  
  # Expectation value for the time to find a target at this point
  if p_prior is None:
    p_prior = np.ones_like(p_discovery)
    
  maybe_vector_valued = p_prior *  activation_function(p_discovery)
  if np.ndim(maybe_vector_valued) == 3:
    maybe_vector_valued = np.sum(maybe_vector_valued, axis=-1)
  score_map = maybe_vector_valued
  
  return np.sum(score_map)/np.prod(score_map.shape)

def diffuse(array, weight = 0.2):
  return (1.0 - 4*weight - 4*weight**2) * array \
  + weight * np.roll(array, ( 1, 0)) \
  + weight * np.roll(array, (-1, 0)) \
  + weight * np.roll(array, ( 0, 1)) \
  + weight * np.roll(array, ( 0,-1)) \
  + weight**2 * np.roll(array, ( 1, 1)) \
  + weight**2 * np.roll(array, ( 1,-1)) \
  + weight**2 * np.roll(array, (-1, 1)) \
  + weight**2 * np.roll(array, (-1,-1))

def compute_prior(geography, method="diffusion"):
  """
  The estimated probability distribution for targets depending on local geography.
  For now simply concentrated probability around roads. 
  """
  target_probability = np.copy(geography.roads)
  for _ in range(20):
    target_probability = diffuse(target_probability)
  return target_probability
