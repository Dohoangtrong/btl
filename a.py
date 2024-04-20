import pygame
import sys
import random
import math
import time

SIZE = 20
CELL_SIZE = 30
WIDTH, HEIGHT = SIZE*CELL_SIZE, SIZE*CELL_SIZE
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
COLOR = (128,128,0)
graph = [[] for _ in range (SIZE*SIZE)]
FPS = 60

class PrimsRandomized:
    def __init__(self):
        self.row_len = SIZE
        self.total_nodes = SIZE ** 2
        self.TOP = 0
        self.LEFT = 1
        self.BOTTOM = 2
        self.RIGHT = 3

    def prims_mst(self):
        mst = [
            [0, 0, 0, 0] for _ in range(self.total_nodes)
        ]
        to_visit = [node for node in range(self.total_nodes)]
        node = to_visit[0]
        visited = [node]
        to_visit.remove(node)

        while len(to_visit) > 0:
            edges_pool = self.edges_to_unvisited_nodes(visited)
            edge = random.choice(edges_pool)
            node, next_node = edge
            direction = self.get_neighbour_dir(node, next_node)
            mst[node][direction] = 1
            neighbour_dir = self.get_neighbour_dir(next_node, node)
            mst[next_node][neighbour_dir] = 1
            visited.append(next_node)
            to_visit.remove(next_node)

        return mst

    def edges_to_unvisited_nodes(self, visited):
        edges_pool = []

        for node in visited:
            row = node // self.row_len
            col = node % self.row_len
            if row > 0:
                top_node = node - self.row_len
                if top_node not in visited:
                    edges_pool.append((node, top_node))
            if col > 0:
                left_node = node - 1
                if left_node not in visited:
                    edges_pool.append((node, left_node))
            if row < self.row_len - 1:
                bottom_node = node + self.row_len
                if bottom_node not in visited:
                    edges_pool.append((node, bottom_node))
            if col < self.row_len - 1:
                right_node = node + 1
                if right_node not in visited:
                    edges_pool.append((node, right_node))
        return edges_pool

    def get_neighbour_dir(self, node, next_node):
        if node - self.row_len == next_node:
            return self.TOP
        if node - 1 == next_node:
            return self.LEFT
        if node + self.row_len == next_node:
            return self.BOTTOM
        if node + 1 == next_node:
            return self.RIGHT

class RectMaze:

    def __init__(self, screen):
        self.size = SIZE
        self.sideLen = CELL_SIZE
        self.screen = screen

    def create_maze(self):
        pr = PrimsRandomized()
        mst = pr.prims_mst()
        for row in range(self.size):
            for col in range(self.size):
                rowLen = row * self.sideLen
                colLen = col * self.sideLen
                node = row * self.size + col
                if mst[node][pr.TOP] == 0:
                    pygame.draw.line(self.screen, BLACK, (colLen, rowLen), (colLen + self.sideLen, rowLen ))
                else: graph[node].append(node-SIZE)
                if mst[node][pr.RIGHT] == 0:
                    pygame.draw.line(self.screen, BLACK, (colLen + self.sideLen, rowLen), (colLen + self.sideLen, rowLen + self.sideLen))
                else: graph[node].append(node+1)
                if mst[node][pr.BOTTOM] == 0:
                    pygame.draw.line(self.screen, BLACK, (colLen, rowLen + self.sideLen), (colLen + self.sideLen, rowLen + self.sideLen))
                else: graph[node].append(node+SIZE)
                if mst[node][pr.LEFT] == 0:
                    pygame.draw.line(self.screen, BLACK, (colLen, rowLen), (colLen, rowLen + self.sideLen))
                else: graph[node].append(node-1)
                

class DFS:
    def __init__(self, screen, start, end):
        self.screen = screen
        self.start = start
        self.end = end
    
    def run_dfs(self):
        dfs(self.start, self.end, self.screen)
    

def dfs(start, end, screen):
    visited = set() 
    stack = [[0,start]]
    while stack:
        vertex = stack.pop()
        if vertex[1] == end: 
            draw_line(screen,vertex[1],COLOR)
            pygame.display.flip()
            time.sleep(0.05)
            break
        print(vertex[0], vertex[1])
        if vertex[1] not in visited:
            visited.add(vertex[1])
            stack.append([vertex[0],vertex[1]])  
            for neighbor in graph[vertex[1]]:
                if neighbor not in visited:
                    stack.append([vertex[1],neighbor])
            draw_line(screen,stack[-1][0],COLOR)
            pygame.display.flip()
            time.sleep(0.05)
        else:
            draw_line(screen,vertex[1],COLOR)
            pygame.display.flip()
            time.sleep(0.05)
            draw_line(screen,vertex[1],WHITE)
            pygame.display.flip()
            time.sleep(0.05)
            


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("DFS IN MAZE")
    clock = pygame.time.Clock()
    running = True

    screen.fill(WHITE)
    rm = RectMaze(screen)
    rm.create_maze()

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                dfs = DFS(screen, 0, SIZE**2 - 1)
                dfs.run_dfs()
        
        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit()

def draw_line(screen, dot, color):      
    x_dot = dot // SIZE * CELL_SIZE + CELL_SIZE // 2 
    y_dot = dot % SIZE * CELL_SIZE + CELL_SIZE // 2 
    pygame.draw.circle(screen, color, (y_dot, x_dot), 8)

if __name__ == "__main__":
    main()
