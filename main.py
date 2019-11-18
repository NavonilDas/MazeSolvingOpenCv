import cv2
import numpy as np


CELL = 20
img = cv2.imread("maze2.jpg")

# Height & Width
h,w = img.shape[:2]

nrow = h // CELL
ncols = w // CELL
ncells = nrow * ncols

Graph = {}
def Connect(x,y,i,j):
    node1 = x*ncols + y
    node2 = i*ncols + j
    if node1 >= ncells or node2 >= ncells or node1 < 0 or node2 < 0:
        return
    if node1 not in Graph:
        Graph[node1] = []
    Graph[node1].append(node2)

def Bfs():
    visited = [False]*ncells
    visited[0] = True
    queue = []
    queue.append(0)
    parents = [-1]*ncells
    dist = [ncells]*ncells
    dist[0] = 0

    while queue:
        s = queue.pop()

        for i in Graph[s]:
            if not visited[i]:
                queue.append(i)
                visited[i] = True
                parents[i] = s
                dist[i] = 1 + dist[s]
                if i == ncells - 1:
                    return (parents,dist[ncells-1])


# for i in range(CELL,w,CELL):
#     cv2.line(img,(i,0),(i,w),(0,0,255),1)
#     cv2.line(img,(0,i),(w,i),(0,0,255),1)
# tops = np.ones(img.shape,img.dtype)*255

for i in range(0,nrow):
    for j in range(0,ncols):
        x,y = j*20,i*20
        
        top = img[y:(y+2),(x+2):(x+CELL-4)]
        left = img[(y+2):(y+CELL-4),x:(x+2)]
        bottom = img[(y+CELL-4):(y+CELL),(x+2):(x+CELL-4)]
        right = img[(y+2):(y+CELL-4),(x+CELL-4):(x+CELL)]

        if 255 not in cv2.inRange(top,(0,0,0),(1,1,1)):
            Connect(i,j,i-1,j)
            # cv2.rectangle(tops,(x+2,y),(x+CELL-4,y+2),(255,0,0),-1)
        if 255 not in cv2.inRange(left,(0,0,0),(1,1,1)):
            Connect(i,j,i,j-1)
            # cv2.rectangle(tops,(x,y+2),(x+2,y+CELL-4),(255,0,0),-1)
        if 255 not in cv2.inRange(bottom,(0,0,0),(1,1,1)):
            Connect(i,j,i+1,j)
            # cv2.rectangle(tops,(x+2,y+CELL-4),(x+CELL-4,y+CELL),(255,0,0),-1)
        if 255 not in cv2.inRange(right,(0,0,0),(1,1,1)):
            Connect(i,j,i,j+1)
            # cv2.rectangle(tops,(x+CELL-4,y+2),(x+CELL,y+CELL-4),(255,0,0),-1)

par,d = Bfs()

answer = [(nrow-1,ncols-1)]
x = par[ncells-1]
while(par[x] != -1):
    answer.append((x//ncols,x%ncols))
    x = par[x]
answer.append((0,0))
answer.reverse()
print(answer)
print(len(answer))

ansvis = img.copy()
for p in answer:
    x,y = p
    x,y = x*CELL,y*CELL
    x,y = y,x
    cv2.rectangle(ansvis,(x,y),(x+CELL,y+CELL),(255,0,0),-1)

cv2.imshow("original",img)
# cv2.imshow("visualize1",tops)
cv2.imshow("answer",ansvis)
cv2.waitKey(0)