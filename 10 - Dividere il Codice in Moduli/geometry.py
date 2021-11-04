class Triangle:
  
  """A simple class to calculate measurements of a triangle"""
  
  def __init__(self, a, b, c, h):
    
    self.a, self.b, self.c, self.h = a, b, c ,h

    
  def area(self):
    
    """Calculate the triangle's area"""
    
    return float(self.b)*float(self.h)/2.
  
  
  def perimeter(self):
    
    """Calculate the triangle's perimeter"""
    
    return self.a+self.b+self.c
  
  
  
class Square:
  
  """A simple class to calculate measurements of a square"""
  
  def __init__(self, l):
    self.l = l
    
    
  def area(self):
    
    """Calculate the square's area"""
    
    return self.l**2
  
 
  def perimeter(self):
    
    """Calculate the square's perimeter"""
    
    return self.l*4.
  
  
  
class Rectangle:
  
  """A simple class to calculate measurements of a rectangle"""
  
  def __init__(self, b, h):
    self.b, self.h = b, h
    
    
  def area(self):
    
    """Calculate the rectangle's area"""
    
    return self.b*self.h
  
 
  def perimeter(self):
    
    """Calculate the rectangle's perimeter"""
    
    return 2.*(self.b+self.h)