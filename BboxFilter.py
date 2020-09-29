class BboxFilter:

    # grid size in pixels
    # filter size max count
    def __init__(self, grid_sz=30, filter_sz=3):
        self.grid_sz = grid_sz
        self.filter_sz = filter_sz
        self.filter_thresh = round(self.filter_sz/2)
        self.bbox_grid = {}
        self.boxes = {} # lookup table
        self.frame_boxes = set() # collect added h in frame

    def box_hash(self, bbox):
        x1,y1,x2,y2 = bbox
        h = 1000000*round(x1/self.grid_sz) + 10000*round(y1/self.grid_sz) + 100*round(x2/self.grid_sz) + round(y2/self.grid_sz)
        return h

    # def add_boxes(self, bboxes):
                
    #     hs = set() # collect added h

    #     # get new box hashes and add to grid or increment
    #     for b in bboxes:            
    #         h = self.box_hash(b)
    #         hs.add(h)
            
    #         if h in self.bbox_grid:                
    #             if self.bbox_grid[h] < self.filter_sz: # limit to max count
    #                 self.bbox_grid[h] += 1    
    #         else:
    #             self.bbox_grid[h] = 1

    #     boxes = []
    #     for h in self.bbox_grid:
    #         # decrement boxes not in the list min to 0
    #         if h not in hs and self.bbox_grid[h] > 0:
    #             self.bbox_grid[h] -= 1

    #         if self.bbox_grid[h] >= self.filter_thresh:
    #             boxes.append(b) # yes add to returns            

    #     return boxes

    # use add to add boxes individually in concert with 
    # get_boxes 
    def add(self, box, conf, cls):         
        #print("add",box)       
        # get new box hashes and add to grid or increment        
        h = self.box_hash(box)

        if h in self.frame_boxes:
            #print("already", h)
            self.boxes[h] = (box, conf, cls) # last box overrides same box hashes
            self.frame_boxes.add(h)
            return # lets not double count 2 box in same image

        self.boxes[h] = (box, conf, cls) # last box overrides same box hashes #TODO add id
        self.frame_boxes.add(h)

        # store hash count in grid
        if h in self.bbox_grid:                
            if self.bbox_grid[h] < self.filter_sz: # limit to max count
                self.bbox_grid[h] += 1    
        else:
            self.bbox_grid[h] = 1

    def get_boxes(self): # call it at every frame
        bxs = []
        # titerate the whole grid
        for h in self.bbox_grid: 

            # decremnent boxes that are not present in this frame's boxes
            if h not in self.frame_boxes and self.bbox_grid[h] > 0:
                self.bbox_grid[h] -= 1
            
            if self.bbox_grid[h] >= self.filter_thresh:                
                bxs.append(self.boxes[h]) # yes add to returns

        self.frame_boxes = set() 
        return bxs

    # def add_get(self, box, conf, cls):
    #     h = self.box_hash(box)


def test():

    x1 = 1900
    y1 = 1000
    x2 = 1919
    y2 = 1079

    box1 = (x1,y1,x2,y2)
    box2 = (1876, 987, 1910, 1065)

    bboxes = [box1, box2] 

    bbox_filter = BboxFilter(30, 3)
    print(bbox_filter.bbox_grid)
    res = bbox_filter.add_boxes(bboxes)
    print(res)
    print(bbox_filter.bbox_grid)

    res = bbox_filter.add_boxes([[12,34,45,67]])
    print(res)
    print(bbox_filter.bbox_grid)
    res = bbox_filter.add_boxes([[12,34,45,67]])
    print(bbox_filter.bbox_grid)
    print(res)


    # 0 1 1 0 1 1 0 0 0 1 0 0 1 1 0
    # 0 1 2 1 2 3 2 1 0 1 0 0 1 2 1

    # 0 0 1 0 1 1 1 0 0 0 0 0 0 1 0 # 2 
    # 0 1 1 0 1 1 1 1 0 1 0 0 1 1 1 # 1

    # 0 0 1 0 0 1 0 0 0 1 0 0 1 0
def test2():
    bbox_filter = BboxFilter(30, 3)
    print(bbox_filter.bbox_grid)

    bbox_filter.add((1876, 987, 1910, 1065), 0.12, 12)
    bbox_filter.add((12,34,45,67),0.43,8)
    boxes = bbox_filter.get_boxes()
    print("grid",bbox_filter.bbox_grid)
    print(boxes)

    bbox_filter.add((1900, 1000, 1919, 1079), 0.3, 5)
    bbox_filter.add((12,34,45,64),0.32,7)
    boxes = bbox_filter.get_boxes()
    print("grid",bbox_filter.bbox_grid)
    print(boxes)

#test2()


