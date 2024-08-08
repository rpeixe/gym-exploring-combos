import numpy as np

def decode_individual(gene):
    result = []
    moves = gene.split()
    for move in moves:
        result.append(decode_move(move))
    return result

def decode_move(move):
    BUTTONS = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A', None, 'L', 'R']
    result = np.zeros(12, np.float32)
    match move[0]:
        case '1':
            result[BUTTONS.index('DOWN')] = 1
            result[BUTTONS.index('LEFT')] = 1
        case '2':
            result[BUTTONS.index('DOWN')] = 1
        case '3':
            result[BUTTONS.index('DOWN')] = 1
            result[BUTTONS.index('RIGHT')] = 1
        case '4':
            result[BUTTONS.index('LEFT')] = 1
        case '5':
            pass
        case '6':
            result[BUTTONS.index('RIGHT')] = 1
        case '7':
            result[BUTTONS.index('UP')] = 1
            result[BUTTONS.index('LEFT')] = 1
        case '8':
            result[BUTTONS.index('UP')] = 1
        case '9':
            result[BUTTONS.index('UP')] = 1
            result[BUTTONS.index('RIGHT')] = 1
        case _:
            raise ValueError(f'First character {move[0]} of move {move} is not a number')
    
    match move[1:]:
        case '-':
            pass
        case 'lp':
            result[BUTTONS.index('L')] = 1
        case 'mp':
            result[BUTTONS.index('L')] = 1
            result[BUTTONS.index('B')] = 1
        case 'hp':
            result[BUTTONS.index('B')] = 1
        case 'lk':
            result[BUTTONS.index('R')] = 1
        case 'mk':
            result[BUTTONS.index('R')] = 1
            result[BUTTONS.index('A')] = 1
        case 'hk':
            result[BUTTONS.index('A')] = 1
        case _:
            raise ValueError(f'Second part {move[1:]} of move {move} was not recognized')
    return result
