"""
Created 2016.06.27
@author: Kyle Ellefsen

This code solves a puzzle made by Eitan Cher for the International Puzzle Party 2016.  One side of the puzzle is the
Japanese Flag, the other is the Canadian Flag.  There are 10 pieces, which fit into both sides.  There are a very
limited number of ways the pieces can fit together inside either flag.

This code tries to lay down pieces, which can be rotated, translated and flipped, in every possible combination, until
either no piece fits or a gap in the puzzle is created which is smaller than 4 pixels, the size of the smallest piece.
"""

import numpy as np
from scipy.ndimage import label
from scipy import ndimage
import time

maple = np.array(  [[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0],
                    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]], dtype=np.bool)

sun   = np.array(  [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.bool)

pieces = [np.array([[0, 1, 0],
                     [0, 1, 0],
                     [1, 1, 0],
                     [1, 1, 1]], dtype=np.bool),

          np.array([[0, 1],
                    [1, 1],
                    [1, 0],
                    [1, 1],
                    [0, 1]], dtype=np.bool),

          np.array([[1, 1, 0],
                    [0, 1, 0],
                    [0, 1, 1],
                    [0, 0, 1]], dtype=np.bool),

          np.array([[0, 0, 1, 0],
                    [0, 1, 1, 1],
                    [1, 1, 0, 0]], dtype=np.bool),

          np.array([[0, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.bool),

          np.array([[1, 1, 0],
                    [1, 1, 1],
                    [0, 1, 0]], dtype=np.bool),

          np.array([[0, 1, 0],
                     [1, 1, 1],
                     [1, 0, 1]], dtype=np.bool),

           np.array([[1, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]], dtype=np.bool),

           np.array([[0, 1, 0],
                     [1, 1, 1],
                     [0, 0, 1]], dtype=np.bool),

           np.array([[1, 1, 0],
                     [1, 1, 1]], dtype=np.bool)]

NoFlip = [2, 4, 5, 6]  # These pieces don't need to be flipped
NoRotate = [4]  # This piece doesn't need to be rotated.


pieces_rotated_flipped = [[[[] for j in [0,1,2,3]] for i in [0,1]] for p in pieces]
where_pieces           = [[[[] for j in [0,1,2,3]] for i in [0,1]] for p in pieces]
for p, piece in enumerate(pieces):
    for flipped in [0, 1]:
        for rotation in [0, 1, 2, 3]:
            piece2 = np.copy(piece)
            if flipped == 1:
                piece2 = np.fliplr(piece2)
            if rotation == 1:
                piece2 = np.rot90(piece2,1)
            if rotation == 2:
                piece2 = np.rot90(piece2,2)
            if rotation == 3:
                piece2 = np.rot90(piece2,3)
            pieces_rotated_flipped[p][flipped][rotation] = piece2
            where_pieces[p][flipped][rotation] = np.where(piece2)
pieces_size = np.array([np.count_nonzero(piece) for piece in pieces])


def get_possible_piece_coords(piece_idx, maple):
    if piece_idx in NoFlip:
        flip_options = [0]
    else:
        flip_options = [0,1]
    if piece_idx in NoRotate:
        rotate_options=[0]
    else:
        rotate_options = [0,1,2,3]
    piece_coordinates = []
    for flipped in flip_options:
        for rotation in rotate_options:
            piece = pieces_rotated_flipped[piece_idx][flipped][rotation]
            px, py = piece.shape
            piece_pos_x, piece_pos_y = where_pieces[piece_idx][flipped][rotation]
            piece_pos_range = np.arange(len(piece_pos_x))
            for x in np.arange(maple.shape[0]-px+1):
                for y in np.arange(maple.shape[1]-py+1):
                    piece_fits = True
                    for v in piece_pos_range:
                        if not maple[x+piece_pos_x[v], y+piece_pos_y[v]]:
                            piece_fits = False
                            break
                    if piece_fits:
                        new_maple=np.copy(maple)
                        new_maple[piece_pos_x+x, piece_pos_y+y] = 0
                        if np.all(np.all(new_maple==False)):
                            print('Solution Found.')
                            piece_coordinates.append([x, y, rotation, flipped, new_maple])
                        else:
                            contiguous = isContiguous(new_maple)
                            if contiguous:
                                piece_coordinates.append([x, y, rotation, flipped, new_maple] )
    return piece_coordinates


def isContiguous(maple):
    labeled_array, nLabels = label(maple)
    if nLabels==1:
        return True
    sizes = ndimage.sum(maple, labeled_array, range(1, nLabels + 1)).astype(np.uint8)
    if np.min(sizes) >= 4:
        return True
    else:
        return False


def get_descendants(maple, curr_piece=0,skip=0):
    solution_found = False
    next_piece = curr_piece+1
    piece_coordinates = get_possible_piece_coords(curr_piece, maple)
    if curr_piece == 0:
        print('Total number of positions for first piece: {}'.format(len(piece_coordinates)))
    for i, coord in enumerate(piece_coordinates[skip:]):
        if curr_piece ==9:
            print('Solved!')
            return [coord]
        if curr_piece == 0:
            print(i+skip)
        new_maple = coord[-1]
        lower_coord = get_descendants(new_maple, next_piece)
        if lower_coord is not None:
            coord=[coord]
            coord.extend(lower_coord)
            return coord
    return None


def print_maple(maple):
    for j in maple:
        for i in j:
            if i:
                print('x ', end="")
            else:
                print('  ', end="")
        print('')

def print_solution(maple, pieces_solution):
    solution = -1*np.ones_like(pieces_solution[0][-1], dtype=np.int)
    solution[np.where(maple-pieces_solution[0][-1])] = 0
    for i, piece in enumerate(pieces_solution):
        bool_array = piece[-1]
        solution[np.where(bool_array)] = i+1
    for i in np.arange(solution.shape[0]):
        for j in np.arange(solution.shape[1]):
            n = solution[i,j]
            if n==-1:
                print('   ', end='')
            else:
                print('{:3}'.format(n) ,end='')
        print()

side = 'MAPLE LEAF'
if side == 'MAPLE LEAF':
    solution = get_descendants(maple)
    print_solution(maple, solution)
elif side == 'RISING SUN':
    solution = get_descendants(sun, skip=78)
    print_solution(sun, solution)
