import pygame
import sys
import heapq
from collections import deque

pygame.init()

screen_width, screen_height = 1000, 700
grid_width = 700
panel_width = screen_width - grid_width
screen = pygame.display.set_mode((screen_width, screen_height))# creates a window
pygame.display.set_caption("Let's play!")

WHITE = (255, 255, 255)  # Background colour
BLACK = (0, 0, 0)        # Barrier colour
ORANGE = (255, 165, 0)   # Start colour
TURQUOISE = (64, 224, 208)  # End colour
RED = (255, 99, 71)      # Closed (Visited) colour
GREEN = (60, 179, 113)   # Open (In Process) colour
PURPLE = (128, 0, 128)   # Path colour (Purple)
GREY = (192, 192, 192)   # Grid line colour
GRAY = (227, 189, 52) # Button's colour
BLUE = (217, 104, 17) # Button's colour when pressed
PURPLE_BG = (67, 24, 120)  # Halloween-themed purple background color

font = pygame.font.SysFont('Helvetica', 36, bold=True)
button_width, button_height = 200, 80
button_spacing = 20

buttons = [
    pygame.Rect(
        grid_width + (panel_width - button_width) // 2, # Centered in panel
        50 + i * (button_height + button_spacing), # Spacing between buttons
        button_width, 
        button_height
    )
    for i in range(6)
]

button_texts = ["DFS", "BFS", "UCS", "Dijkstra", "A*", "Clear"]

pumpkin_img = pygame.image.load('pumpkin.png')
ghost_img = pygame.image.load('ghost.png')

cell_size = grid_width // 50 
pumpkin_img = pygame.transform.scale(pumpkin_img, (cell_size, cell_size))
ghost_img = pygame.transform.scale(ghost_img, (cell_size, cell_size))


class Cell:
    def __init__(self, row, col, width, total_rows):
        self.row = row
        self.col = col
        self.x = row * width
        self.y = col * width
        self.colour = PURPLE_BG
        self.width = width
        self.total_rows = total_rows
        self.neighbours = []
        self.cost = 1

        # Flags for image usage
        self.is_pumpkin = False
        self.is_ghost = False

    def get_pos(self):
        return self.row, self.col

    def is_barrier(self):
        return self.colour == BLACK

    def is_start(self):
        return self.colour == ORANGE

    def is_end(self):
        return self.colour == TURQUOISE 

    def make_start(self):
        self.colour = ORANGE
        self.is_pumpkin = True
        self.is_ghost = False

    def make_end(self):
        self.colour = TURQUOISE
        self.is_pumpkin = False
        self.is_ghost = True

    def make_barrier(self):
        self.colour = BLACK
        self.cost = float('inf')

    def make_open(self):
        self.colour = GREEN
        self.cost = 1

    def make_closed(self):
        self.colour = RED

    def reset(self):
        self.colour = PURPLE_BG
        self.cost = 1
        self.is_pumpkin = False
        self.is_ghost = False

    def make_path(self):
        self.colour = PURPLE

    def draw(self, win):
       ## pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.width)) # draw the cell as a rectangle: parameters: surface win, colour of rectangle, rect-position and size of the rectangle(x,y,width,height), width-optional for border
       if self.is_pumpkin:
            win.blit(pumpkin_img, (self.x, self.y))
       elif self.is_ghost:
            win.blit(ghost_img, (self.x, self.y))
       else:
            pygame.draw.rect(win, self.colour, (self.x, self.y, self.width, self.width))

    def update_neighbours(self, grid):
        self.neighbours = []
        if self.row < self.total_rows - 1 and not grid[self.row + 1][self.col].is_barrier():  # Down
            self.neighbours.append(grid[self.row + 1][self.col])
        if self.row > 0 and not grid[self.row - 1][self.col].is_barrier():  # Up
            self.neighbours.append(grid[self.row - 1][self.col])
        if self.col < self.total_rows - 1 and not grid[self.row][self.col + 1].is_barrier():  # Right
            self.neighbours.append(grid[self.row][self.col + 1])
        if self.col > 0 and not grid[self.row][self.col - 1].is_barrier():  # Left
            self.neighbours.append(grid[self.row][self.col - 1])
    
    def __lt__(self, other):
        return self.cost < other.cost  # less-than, compare based on cost self(current) with other(neighbour)

def make_grid(rows, width):
    grid = []
    gap = width // rows
    for i in range(rows):
        grid.append([])
        for j in range(rows):
            cell = Cell(i, j, gap, rows)
            grid[i].append(cell)
    return grid

def draw_grid(win, rows, width):
    gap = width // rows
    for i in range(rows):
        pygame.draw.line(win, GREY, (0, i * gap), (width, i * gap)) # surface, colour, start position and end position -- for rows
        for j in range(rows):
            pygame.draw.line(win, GREY, (j * gap, 0), (j * gap, width)) # for columns

def draw_buttons(win):
    pygame.draw.rect(win, PURPLE_BG, (grid_width, 0, panel_width, screen_height))
    for i, button_rect in enumerate(buttons):
        if button_rect.collidepoint(pygame.mouse.get_pos()):
            pygame.draw.rect(win, BLUE, button_rect)
        else:
            pygame.draw.rect(win, GRAY, button_rect)
        text_surface = font.render(button_texts[i], True, BLACK)
        win.blit(text_surface, text_surface.get_rect(center=button_rect.center)) #  draw images or surfaces onto the screen (or another surface)

def draw(win, grid, rows, width):
    win.fill(PURPLE_BG)
    for row in grid:
        for cell in row:
            cell.draw(win)
    draw_grid(win, rows, width)
    draw_buttons(win)
    pygame.display.update()

def get_clicked_pos(pos, rows, width): # converts a pixel position (pos) into grid coordinates (row, col)
    gap = width // rows
    y, x = pos
    row = y // gap
    col = x // gap
    return row, col


def alertCycle(button_label, grid, start, end):
    if button_label == "Clear": 
        for row in grid:
            for cell in row:
                cell.reset()
        return None, None
    elif button_label == "DFS":
        if start and end:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid)
            dfs(lambda: draw(screen, grid, 50, grid_width), grid, start, end)
    elif button_label == "BFS":
        if start and end:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid) 
            bfs(lambda: draw(screen, grid, 50, grid_width), grid, start, end) # lambda is a way to create an anonymous function
    elif button_label == "UCS":
        if start and end:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid)
            ucs(lambda: draw(screen, grid, 50, grid_width), grid, start, end)
    elif button_label == "Dijkstra":
        if start and end:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid)
            dijkstra(lambda: draw(screen, grid, 50, grid_width), grid, start, end)
    elif button_label == "A*":
        if start and end:
            for row in grid:
                for cell in row:
                    cell.update_neighbours(grid)
            a_star(lambda: draw(screen, grid, 50, grid_width), grid, start, end)
    return start, end


def dfs(draw, grid, start, end):
    stack = [start]
    visited = set()
    came_from = {}

    while stack:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = stack.pop()

        if current not in visited:
            visited.add(current)

            if current != start and current != end:
                current.make_open()

            if current == end:
                path = []
                while current in came_from:
                    path.append(current.get_pos())
                    current = came_from[current]
                path.append(start.get_pos())

                path.reverse()
                print("Path:")
                print("Start:")
                for coord in path:
                    print(f"  {coord},") #formatted string literal(%d, %s) 
                print("Goal")
                print(f"Length: {len(path) - 1} steps")

                for p in path:
                    if p != start.get_pos() and p != end.get_pos():
                        grid[p[0]][p[1]].make_path() #accesses the cell in the grid at the row and column specified by the tuple p(draw the path)
                draw()
                return True

            for neighbour in current.neighbours:
                if neighbour not in visited and not neighbour.is_barrier():
                    came_from[neighbour] = current
                    stack.append(neighbour)
                    if neighbour != end:
                        neighbour.make_open()

            draw()

            if current != start and current != end:
                current.make_closed()

    return False


def bfs(draw, grid, start, end):
    queue = deque([start])
    visited = set()
    came_from = {}

    while queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = queue.popleft()

        if current not in visited:
            visited.add(current)

            if current != start and current != end:
                current.make_open()

            if current == end:
                path = []
                while current in came_from:
                    path.append(current.get_pos())
                    current = came_from[current]
                path.append(start.get_pos())

                path.reverse()
                print("Path:")
                print("Start:")
                for coord in path:
                    print(f"  {coord},")
                print("Goal")
                print(f"Length: {len(path) - 1} steps")

                for p in path:
                    if p != start.get_pos() and p != end.get_pos():
                        grid[p[0]][p[1]].make_path()
                draw()
                return True

            for neighbour in current.neighbours:
                if neighbour not in visited and not neighbour.is_barrier():
                    came_from[neighbour] = current
                    queue.append(neighbour)
                    if neighbour != end:
                        neighbour.make_open()

            draw()

            if current != start and current != end:
                current.make_closed()

    return False


def ucs(draw, grid, start, end):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start)) #min-heap, ensuring we always explore the least costly node next
    visited = set()
    came_from = {}
    cost_so_far = {start: 0}

    while priority_queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current_cost, current = heapq.heappop(priority_queue) #retrieves and removes the node with the lowest cost from the priority queue

        if current in visited:
            continue

        visited.add(current)

        if current != start and current != end:
            current.make_open()

        if current == end:
            path = []
            while current in came_from:
                path.append(current.get_pos())
                current = came_from[current]
            path.append(start.get_pos())

            path.reverse()
            print("Path:")
            print("Start:")
            for coord in path:
                print(f"  {coord},")
            print("Goal")
            print(f"Length: {len(path) - 1} steps")

            for p in path:
                if p != start.get_pos() and p != end.get_pos():
                    grid[p[0]][p[1]].make_path()
            draw()
            return True

        for neighbour in current.neighbours:
            new_cost = current_cost + 1
            if neighbour not in visited and not neighbour.is_barrier():
                if neighbour not in cost_so_far or new_cost < cost_so_far[neighbour]: #checks if we have either never seen this neighbor before or if we have found a cheaper path to it
                    cost_so_far[neighbour] = new_cost
                    came_from[neighbour] = current
                    heapq.heappush(priority_queue, (new_cost, neighbour))
                    if neighbour != end:
                        neighbour.make_open()

        draw()

        if current != start and current != end:
            current.make_closed()

    return False


def dijkstra(draw, grid, start, end):
    priority_queue = []
    heapq.heappush(priority_queue, (0, start))
    visited = set()
    came_from = {}
    g_score = {cell: float('inf') for row in grid for cell in row} # all cells are unreachable from the starting cell
    g_score[start] = 0

    while priority_queue:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current_cost, current = heapq.heappop(priority_queue) 

        if current in visited:
            continue

        visited.add(current)

        if current != start and current != end:
            current.make_open()

        if current == end:
            path = []
            while current in came_from:
                path.append(current.get_pos())
                current = came_from[current]
            path.append(start.get_pos())

            path.reverse()
            print("Path:")
            print("Start:")
            for coord in path:
                print(f"  {coord},")
            print("Goal")
            print(f"Length: {len(path) - 1} steps")

            for p in path:
                if p != start.get_pos() and p != end.get_pos():
                    grid[p[0]][p[1]].make_path()
            draw()
            return True

        for neighbour in current.neighbours:
            tentative_g_score = g_score[current] + 1
            if neighbour not in visited and not neighbour.is_barrier():
                if tentative_g_score < g_score[neighbour]:
                    came_from[neighbour] = current
                    g_score[neighbour] = tentative_g_score
                    heapq.heappush(priority_queue, (tentative_g_score, neighbour))
                    if neighbour != end:
                        neighbour.make_open()

        draw()

        if current != start and current != end:
            current.make_closed()

    return False


def heuristic(a, b):
    # Using Manhattan distance as the heuristic
    return abs(a.row - b.row) + abs(a.col - b.col)

def a_star(draw, grid, start, end):
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    
    g_score = {cell: float('inf') for row in grid for cell in row}
    g_score[start] = 0
    
    f_score = {cell: float('inf') for row in grid for cell in row}
    f_score[start] = heuristic(start, end)

    open_set_hash = {start}

    while open_set:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        current = heapq.heappop(open_set)[1]
        open_set_hash.remove(current)

        if current == end:
            path = []
            while current in came_from:
                path.append(current.get_pos())
                current = came_from[current]
            path.append(start.get_pos())

            path.reverse()
            print("Path:")
            print("Start:")
            for coord in path:
                print(f"  {coord},")
            print("Goal")
            print(f"Length: {len(path) - 1} steps")

            for p in path:
                if p != start.get_pos() and p != end.get_pos():
                    grid[p[0]][p[1]].make_path()
            draw()
            return True

        for neighbour in current.neighbours:
            if neighbour.is_barrier():
                continue

            tentative_g_score = g_score[current] + 1

            if tentative_g_score < g_score[neighbour]:
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g_score
                f_score[neighbour] = tentative_g_score + heuristic(neighbour, end)

                if neighbour not in open_set_hash:
                    heapq.heappush(open_set, (f_score[neighbour], neighbour))
                    open_set_hash.add(neighbour)

                    if neighbour != end:
                        neighbour.make_open()

        draw()

        if current != start and current != end:
            current.make_closed()

    return False


def main(screen, grid_width):
    ROWS = 50
    grid = make_grid(ROWS, grid_width)
    start = None
    end = None
    run = True

    while run:
        draw(screen, grid, ROWS, grid_width)

        for event in pygame.event.get(): # checks for events like quitting the game or mouse clicks
            if event.type == pygame.QUIT:
                run = False
            
            if pygame.mouse.get_pressed()[0]:  # Left click
                pos = pygame.mouse.get_pos()
                if pos[0] < grid_width: # Only handle clicks within the grid area
                    row, col = get_clicked_pos(pos, ROWS, grid_width)
                    cell = grid[row][col]
                    if not start and cell != end:
                        start = cell
                        start.make_start()
                    elif not end and cell != start:
                        end = cell
                        end.make_end()
                    elif cell != end and cell != start:
                        cell.make_barrier()
                else:                  # Handle clicks on the buttons
                    for i, button_rect in enumerate(buttons):
                        if button_rect.collidepoint(pos):
                            start, end = alertCycle(button_texts[i], grid, start, end)

            elif pygame.mouse.get_pressed()[2]:  # Right click to reset
                pos = pygame.mouse.get_pos()
                if pos[0] < grid_width:
                    row, col = get_clicked_pos(pos, ROWS, grid_width)
                    cell = grid[row][col]
                    cell.reset()
                    if cell == start:
                        start = None
                    elif cell == end:
                        end = None

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:  # Press 'C' to clear the grid
                    start = None
                    end = None
                    grid = make_grid(ROWS, grid_width)

    pygame.quit()


main(screen, grid_width)