class Terminal:
    def __init__(self, name, x=0, y=0):
        self.name = name
        self.x = x
        self.y = y

    def setX(self, x):
        self.x = x

    def setY(self, y):
        self.y = y


class Block:
    def __init__(self, name, type, width, height):
        self.w = width
        self.h = height
        self.x = 0
        self.y = 0
        self.r = False
        self.type = type
        self.t = Terminal(name)
        self.area = self.w * self.h

    def setX(self, x):
        self.x = x
        self.t.x = self.x + (self.h/2 if self.r else self.w/2)

    def setY(self, y):
        self.y = y
        self.t.y = self.y + (self.w/2 if self.r else self.h/2)

    def setWidth(self, w):
        self.w = w
        self.t.x = self.x + (self.h / 2 if self.r else self.w / 2)

    def setHeight(self, h):
        self.h = h
        self.t.y = self.y + (self.w / 2 if self.r else self.h / 2)

    def setRotate(self, r):
        self.r = r
        self.t.x = self.x + (self.h / 2 if self.r else self.w / 2)
        self.t.y = self.y + (self.w / 2 if self.r else self.h / 2)



class SoftBlock(Block):

    def __init__(self, name=None, type=None, width=1, height=1, area=1, min_ratio=1, max_ration=1):
        super().__init__(name, type, width, height)
        self.area = area
        self.min_ratio = min_ratio
        self.max_ratio = max_ration


class Net:
    def __init__(self):
        self.terminalList = []

    def wirelength(self):
        x = 10000000; X = 0; y = 100000000; Y = 0;
        for t in self.terminalList:
            if t.x < x:
                x = t.x
            if t.x > X:
                X = t.x
            if t.y < y:
                y = t.y
            if t.y > Y:
                Y = t.y
        return X-x+Y-y
