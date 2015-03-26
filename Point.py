#Defines a simple (x,y) coordinate in a 2D matrix
class Point():    
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    def move_right(self):
        self.y += 1
    
    def move_left(self):
        self.y -= 1
    
    def move_up(self):
        self.x -= 1
    
    def move_down(self):
        self.x += 1
    
    def __repr__(self):
        return "(%i, %i)"%(self.x, self.y)


