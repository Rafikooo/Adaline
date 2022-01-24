# import pygame
# from button import button
from copy import deepcopy

from adaline import Adaline, fourier_transform
from dane import *

# gridDisplay = pygame.display.set_mode((280, 400))
# pygame.display.get_surface().fill((200, 200, 200))

training_data = learning_data[:, :49]

perceptrons = []
for i in range(10):
  perceptrons.append(Adaline(49))

def train():
    for i in range(10):
      perceptrons[i].train(training_data, labels[i])
    print("Your network is a good student!")


train()


#
# while play:
#     for event in pygame.event.get():
#         if event.type == pygame.QUIT:
#             pygame.quit()
#
#         if event.type == pygame.MOUSEBUTTONDOWN:
#             #mat_set = matrix1
#             mouse_pos = event.pos  # gets mouse position
#
#             # checks if mouse position is over the button
#             if mybutton1.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx1)
#
#             elif mybutton2.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx2)
#
#             elif mybutton3.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx3)
#
#             elif mybutton4.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx4)
#
#             elif mybutton5.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx5)
#
#             elif mybutton6.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx6)
#
#             elif mybutton7.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx7)
#
#             elif mybutton8.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx8)
#
#             elif mybutton9.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx9)
#
#             elif mybutton0.isOver(mouse_pos):
#                 mat_set = deepcopy(matrixx0)
#
#             elif train_button.isOver(mouse_pos):
#                 train()
#
#             elif left_button.isOver(mouse_pos):
#                 mat_set = left(mat_set)
#
#             elif right_button.isOver(mouse_pos):
#                 mat_set = right(mat_set)
#
#
#             if rand_button.isOver(mouse_pos):
#                 mat_set = rand_active(mat_set)
#
#
#
#             visualizeGrid(mat_set)
#             matrix_new = flatten(mat_set)
#             data = np.array(matrix_new)
#
#             x = np.concatenate([data, fourier_transform(data)])
#
#             max_sum = (0, 0)
#             for i in range(10):
#                 sum = perceptrons[i].output(x)
#                 if sum > max_sum[1]:
#                     max_sum = (i, sum)
#
#             result_button = button((255, 123, 166), 160, 360, grid_node_width, grid_node_height, str(max_sum[0]))
#
#
#         result_button.draw(gridDisplay)
#         pygame.display.update()
#
#
#         # pygame_widgets.update(event)
#     # pygame_widgets.update(event)
